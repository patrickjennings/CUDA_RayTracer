/*
 * Modified by:
 * Patrick Jennings
 * Stevie Frederick
 *
 */


// -----------------------------------------------------------------
// Simple cuda ray tracing tutorial
// Written by Peter Trier 
// Alexandra Institute august 2009 
//
//
// -----------------------------------------------------------------


// includes --------------------------------------
#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#pragma warning(disable:4996)
#endif

// includes, GL
#include <GL/glew.h>
#include <GL/glut.h>

// cuda includes
#include <cuda_runtime_api.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cutil_gl_error.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <cutil_math.h>



// std
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;
#include <stdio.h>  //added


// the interface between C++ and CUDA -----------
// the implementation of RayTraceImage is in the 
// "raytracer.cu" file
extern "C" void RayTraceImage(unsigned int *pbo_out, int w, int h, int number_of_triangles,
							  float3 a, float3 b, float3 c, 
							  float3 campos,
							  float3 light_pos,
							  float3 light_color,
							  float3 scene_aabbox_min , float3 scene_aabbox_max);

// a method for binding the loaded triangles to a Cuda texture
// the implementation of bindTriangles is in the 
// "raytracer.cu" file
extern "C" void bindTriangles(float *dev_triangle_p, unsigned int number_of_triangles);


// Obj loader ------------------------------------
struct TriangleFace 
{
	int v[3]; // vertex indices
};

struct TriangleMesh
{
	vector<float3> verts;
	vector<TriangleFace> faces;
	float3 bounding_box[2];
};

// Globals ---------------------------------------
unsigned int window_width  = 800;
unsigned int window_height = 600;
unsigned int image_width   = 1600;
unsigned int image_height  = 1200;
float delta_t = 0;

GLuint pbo;               // this pbo is used to connect CUDA and openGL
GLuint result_texture;    // the ray-tracing result is copied to this openGL texture
TriangleMesh mesh;

TriangleMesh ground;
TriangleMesh sphere;
TriangleMesh object;
int total_number_of_triangles = 0;

float *dev_triangle_p; // the cuda device pointer that points to the uploaded triangles


// Camera parameters -----------------------------
float3 a; float3 b; float3 c; 
float3 campos; 
float camera_rotation = 0;
float camera_distance = 75;
float camera_height = 25;
bool animate = true;

// Scene bounding box ----------------------------
float3 scene_aabbox_min;
float3 scene_aabbox_max;

float light_x = -23;
float light_y = 25;
float light_z = 3;
float light_color[3] = {1,1,1};

// mouse controls --------------------------------
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
bool left_down  = false;
bool right_down = false;

// function declaration --------------------------
bool initGL();
bool initCUDA( int argc, char **argv);
void initCUDAmemory();
void loadObj(const std::string filename, TriangleMesh &mesh);
void display();
void reshape(int width, int height);
void keyboard(unsigned char key, int x, int y);
void SpecialKey(int key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void rayTrace();


bool initGL()
{
	glewInit();
	if (! glewIsSupported
		(
		"GL_VERSION_2_0 " 
		"GL_ARB_pixel_buffer_object "
		"GL_EXT_framebuffer_object "
		)) 
	{
			fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
			fflush(stderr);
			return CUTFalse;
	}

	// init openGL state
	glClearColor(0, 0, 0, 1.0);
	glDisable(GL_DEPTH_TEST);

	// view-port
	glViewport(0, 0, window_width, window_height);

	return true;
}

bool initCUDA( int argc, char **argv)
{

	if ( cutCheckCmdLineFlag(argc, (const char **)argv, "device"))
	{
		cutilGLDeviceInit(argc, argv);
	}
	else 
	{
		cudaGLSetGLDevice (cutGetMaxGflopsDeviceId() );
	}
	
	return true;
}

void initCUDAmemory()
{
	// initialize the PBO for transferring data from CUDA to openGL
	unsigned int num_texels = image_width * image_height;
	unsigned int size_tex_data = sizeof(GLubyte) * num_texels * 4;
	void *data = malloc(size_tex_data);

	// create buffer object
	glGenBuffers(1, &pbo);
	glBindBuffer(GL_ARRAY_BUFFER, pbo);
	glBufferData(GL_ARRAY_BUFFER, size_tex_data, data, GL_DYNAMIC_DRAW);
	free(data);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// register this buffer object with CUDA
	cutilSafeCall(cudaGLRegisterBufferObject(pbo));
	CUT_CHECK_ERROR_GL();

	// create the texture that we use to visualize the ray-tracing result
	glGenTextures(1, &result_texture);
	glBindTexture(GL_TEXTURE_2D, result_texture);

	// set basic parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	// buffer data
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_width, image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	CUT_CHECK_ERROR_GL();

	// next we load a simple obj file and upload the triangles to an 1D texture.
	loadObj("data/cube.obj",mesh);
	loadObj("data/sphere.obj",sphere);

	vector<float4> triangles;

	for(unsigned int i = 0; i < mesh.faces.size(); i++)
	{
		float3 v0 = mesh.verts[mesh.faces[i].v[0]-1];
		float3 v1 = mesh.verts[mesh.faces[i].v[1]-1];
		float3 v2 = mesh.verts[mesh.faces[i].v[2]-1];
		triangles.push_back(make_float4(v0.x,v0.y,v0.z,0));
		triangles.push_back(make_float4(v1.x-v0.x, v1.y-v0.y, v1.z-v0.z,0)); // notice we store the edges instead of vertex points, to save some calculations in the 
		triangles.push_back(make_float4(v2.x-v0.x, v2.y-v0.y, v2.z-v0.z,0)); // ray triangle intersection test.
	}

	for(unsigned int i = 0; i < sphere.faces.size(); i++)
	{
		float3 v0 = sphere.verts[sphere.faces[i].v[0]-1];
		float3 v1 = sphere.verts[sphere.faces[i].v[1]-1];
		float3 v2 = sphere.verts[sphere.faces[i].v[2]-1];
		triangles.push_back(make_float4(v0.x,v0.y,v0.z,0));
		triangles.push_back(make_float4(v1.x-v0.x, v1.y-v0.y, v1.z-v0.z,1)); // notice we store the edges instead of vertex points, to save some calculations in the 
		triangles.push_back(make_float4(v2.x-v0.x, v2.y-v0.y, v2.z-v0.z,0)); // ray triangle intersection test.
	}

	cout << "total number of triangles check:" << mesh.faces.size() + sphere.faces.size() << " == " << triangles.size()/3 << endl;

	size_t triangle_size = triangles.size() * sizeof(float4);
	total_number_of_triangles = triangles.size()/3;

	if(triangle_size > 0)
	{
		cutilSafeCall( cudaMalloc((void **)&dev_triangle_p, triangle_size));
		cudaMemcpy(dev_triangle_p,&triangles[0],triangle_size,cudaMemcpyHostToDevice);
		bindTriangles(dev_triangle_p, total_number_of_triangles);
	}

	scene_aabbox_min = mesh.bounding_box[0];
	scene_aabbox_max = mesh.bounding_box[1];

	scene_aabbox_min.x = min(scene_aabbox_min.x,sphere.bounding_box[0].x);
	scene_aabbox_min.y = min(scene_aabbox_min.y,sphere.bounding_box[0].y);
	scene_aabbox_min.z = min(scene_aabbox_min.z,sphere.bounding_box[0].z);

	scene_aabbox_max.x = max(scene_aabbox_max.x,sphere.bounding_box[1].x);
	scene_aabbox_max.y = max(scene_aabbox_max.y,sphere.bounding_box[1].y);
	scene_aabbox_max.z = max(scene_aabbox_max.z,sphere.bounding_box[1].z);


}

// Callback function called by GLUT when window size changes
void reshape(int width, int height)
{
	// Set OpenGL view port and camera
	glViewport(0, 0, width, height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0f, (double)width/height, 0.1, 100);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}



void SpecialKey(int key, int x, int y)
{
	switch(key)
	{
	case GLUT_KEY_F1:
		
		break;
	
	};
}

void updateCamera()
{
	
	campos = make_float3(cos(camera_rotation)*camera_distance,camera_height,-sin(camera_rotation)*camera_distance);
	float3 cam_dir = -campos;
	cam_dir = normalize(cam_dir);
	float3 cam_up  = make_float3(0,1,0);
	float3 cam_right = cross(cam_dir,cam_up);
	cam_right = normalize(cam_right);

	//cam_up = -cross(cam_dir,cam_right);
	cam_up = cross(cam_dir,cam_right);
	cam_up = -cam_up;
	cam_up = normalize(cam_up);
	
	float FOV = 60.0f;
	float theta = (FOV*3.1415*0.5) / 180.0f;
	float half_width = tanf(theta);
	float aspect = (float)image_width / (float)image_height;

	float u0 = -half_width * aspect;
	float v0 = -half_width;
	float u1 =  half_width * aspect;
	float v1 =  half_width;
	float dist_to_image = 1;

	a = (u1-u0)*cam_right;
	b = (v1-v0)*cam_up;
	c = campos + u0*cam_right + v0*cam_up + dist_to_image*cam_dir;
	
	if(animate)
	camera_rotation += 0.25 * delta_t;
	
}


void rayTrace()
{
	unsigned int* out_data;
	cutilSafeCall(cudaGLMapBufferObject( (void**)&out_data, pbo));

	RayTraceImage(out_data, image_width, image_height,total_number_of_triangles, 
		a, b, c, 
		campos, 
		make_float3(light_x,light_y,light_z),
		make_float3(light_color[0],light_color[1],light_color[2]),
		scene_aabbox_min , scene_aabbox_max);

	cutilSafeCall(cudaGLUnmapBufferObject( pbo));

	// download texture from destination PBO
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glBindTexture(GL_TEXTURE_2D, result_texture);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image_width, image_height, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	CUT_CHECK_ERROR_GL();
}

// display image to the screen as textured quad
void displayTexture()
{
	// render a screen sized quad
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);
	glEnable(GL_TEXTURE_2D);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

	glMatrixMode( GL_MODELVIEW);
	glLoadIdentity();

	glViewport(0, 0, window_width, window_height);

	glBegin(GL_QUADS);
	glTexCoord2f(0.0, 0.0); glVertex3f(-1.0, -1.0, 0.5);
	glTexCoord2f(1.0, 0.0); glVertex3f(1.0, -1.0, 0.5);
	glTexCoord2f(1.0, 1.0); glVertex3f(1.0, 1.0, 0.5);
	glTexCoord2f(0.0, 1.0); glVertex3f(-1.0, 1.0, 0.5);
	glEnd();

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();

	glDisable(GL_TEXTURE_2D);
	CUT_CHECK_ERROR_GL();
}

void display()
{
	//update the delta time for animation
	static int lastFrameTime = 0;

	if (lastFrameTime == 0)
	{
		lastFrameTime = glutGet(GLUT_ELAPSED_TIME);
	}

	int now = glutGet(GLUT_ELAPSED_TIME);
	int elapsedMilliseconds = now - lastFrameTime;
	delta_t = elapsedMilliseconds / 1000.0f;
	lastFrameTime = now;

	updateCamera();

	glClearColor(0,0,0,0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	rayTrace();
	displayTexture();

	glutSwapBuffers();
	glutPostRedisplay();
}

void keyboard(unsigned char key, int /*x*/, int /*y*/)
{

	switch(key) 
	{
	case ' ':
		animate = !animate;
		break;
	case(27) :
		exit(0);
	}
}

int main(int argc, char** argv)
{

	// Create GL context
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(window_width,window_height);
	glutCreateWindow("Alexandra Institute ray-tracing tutorial");

	// initialize GL
	if(CUTFalse == initGL())
	{
		return 0;
	}

	// initialize CUDA
	if(CUTFalse == initCUDA(argc,argv))
	{
		return 0;
	}

	initCUDAmemory();

	// register callbacks
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutSpecialUpFunc(SpecialKey);
	
	glutReshapeFunc(reshape);

	// start rendering main-loop
	glutMainLoop();
	cudaThreadExit();
	
	cutilExit(argc, argv);

	return 0;
}

// load a simple obj file without normals or tex-coords
void loadObj( const std::string filename, TriangleMesh &mesh )
{

	std::ifstream in(filename.c_str());

	if(!in.good())
	{
		cout  << "ERROR: loading obj:(" << filename << ") file is not good" << "\n";
		exit(0);
	}

	char buffer[256], str[255];
	float f1,f2,f3;

	while(!in.getline(buffer,255).eof())
	{
		buffer[255]='\0';

		//sscanf_s(buffer,"%s",str,255);
		sscanf(buffer,"%s",str);

		// reading a vertex
		if (buffer[0]=='v' && (buffer[1]==' '  || buffer[1]==32) )
		{
			if ( sscanf(buffer,"v %f %f %f",&f1,&f2,&f3)==3)
			{
				mesh.verts.push_back(make_float3(f1,f2,f3));
			}
			else
			{
				cout << "ERROR: vertex not in wanted format in OBJLoader" << "\n";
				exit(-1);
			}
		}
		// reading FaceMtls 
		else if (buffer[0]=='f' && (buffer[1]==' ' || buffer[1]==32) )
		{
			TriangleFace f;
			int nt = sscanf(buffer,"f %d %d %d",&f.v[0],&f.v[1],&f.v[2]);
			if( nt!=3 )
			{
				cout << "ERROR: I don't know the format of that FaceMtl" << "\n";
				exit(-1);
			}
			
			mesh.faces.push_back(f);
		}
	}

	// calculate the bounding box
	mesh.bounding_box[0] = make_float3(1000000,1000000,1000000);
	mesh.bounding_box[1] = make_float3(-1000000,-1000000,-1000000);
	for(unsigned int i = 0; i < mesh.verts.size(); i++)
	{
		//update min value
		mesh.bounding_box[0].x = min(mesh.verts[i].x,mesh.bounding_box[0].x);
		mesh.bounding_box[0].y = min(mesh.verts[i].y,mesh.bounding_box[0].y);
		mesh.bounding_box[0].z = min(mesh.verts[i].z,mesh.bounding_box[0].z);

		//update max value
		mesh.bounding_box[1].x = max(mesh.verts[i].x,mesh.bounding_box[1].x);
		mesh.bounding_box[1].y = max(mesh.verts[i].y,mesh.bounding_box[1].y);
		mesh.bounding_box[1].z = max(mesh.verts[i].z,mesh.bounding_box[1].z);

	}

	cout << "obj file loaded: number of faces:" << mesh.faces.size() << " number of vertices:" << mesh.verts.size() << endl;
	cout << "obj bounding box: min:(" << mesh.bounding_box[0].x << "," << mesh.bounding_box[0].y << "," << mesh.bounding_box[0].z <<") max:" 
		<< mesh.bounding_box[1].x << "," << mesh.bounding_box[1].y << "," << mesh.bounding_box[1].z <<")" << endl;


}

