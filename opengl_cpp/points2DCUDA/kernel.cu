#include <glad/glad.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_gl_interop.h>

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "../graphics/shaders/shader.h"
#include "../graphics/cameras/camera2d.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

using namespace std;


// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//
// PASO 1 - DECLARACION DE CONSTANTES, COMO DIMENSIONES , VBO Y VALORES DE CAMARA
//
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

void runKernel(float3* pos, unsigned int mesh_width, float time);
void runTest();
void runCuda(struct cudaGraphicsResource** vbo_resource, float time);
void runDisplay();
void deleteVBO(GLuint* vbo, struct cudaGraphicsResource* vbo_res);

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

const unsigned int window_width = 800;
const unsigned int window_height = 800;
const unsigned int mesh_width = 256;

// camera
Camera2D camera(glm::vec2(0.0f, 0.0f));

// vbo variables
GLuint vbo;
struct cudaGraphicsResource* cuda_vbo_resource;
void* d_vbo_buffer = NULL;

// timing
float deltaTime = 0.0f;	// time between current frame and last frame


__global__ void points2dKernel(float3* pos, unsigned int width, float time)
{
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;

    // calculate uv coordinates
    float u = x / (float)width;
    u = u * 2.0f - 1.0f;
    u = u * 5.0f;

    // calculate simple sine wave pattern
    float freq = 4.0f;
    float w = sinf(u * freq + time) * 0.25f;

    // write output vertex
    pos[x] = make_float3(u, w, 0.0f);
}

int main()
{
    bool cudaTest = false;

    if (cudaTest)
        runTest();
    else
        runDisplay();
}


void runTest()
{
    void* returnData = malloc(mesh_width * sizeof(float));

    // create VBO
    cudaMalloc((void**)&d_vbo_buffer, mesh_width * 3 * sizeof(float));

    // execute the kernel
    runKernel((float3*)d_vbo_buffer, mesh_width, 1.0f);

    cudaDeviceSynchronize();
    cudaMemcpy(returnData, d_vbo_buffer, mesh_width * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_vbo_buffer);
    d_vbo_buffer = NULL;

    free(returnData);
    printf("Test passed");

}

void runKernel(float3* pos, unsigned int mesh_width, float time)
{
    // execute the kernel
    points2dKernel << < 32, 8 >> > (pos, mesh_width, time);
}

void runCuda(struct cudaGraphicsResource** vbo_resource, float time)
{
    // map OpenGL buffer object for writing from CUDA
    float3* dptr;
    cudaGraphicsMapResources(1, vbo_resource, 0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes,
        *vbo_resource);
    //printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);

    runKernel(dptr, mesh_width, time);

    // unmap buffer object
    cudaGraphicsUnmapResources(1, vbo_resource, 0);
}

void runDisplay()
{
    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(window_width, window_height, "2D Points interop CUDA", NULL, NULL);
    if (window == NULL)
    {
        cout << "Failed to create GLFW window" << endl;
        glfwTerminate();
        return;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        cout << "Failed to initialize GLAD" << endl;
        return;
    }

    // build and compile our shader program
    // ------------------------------------
    Shader circleShader("../graphics/shaders/circleTransformShader.vs", "../graphics/shaders/circleTransformShader.fs"); // you can name your shader files however you like

    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------

    unsigned int VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &vbo);
    // bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
    glBindVertexArray(VAO);

    // create buffer object
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    // initialize buffer object
    unsigned int size = mesh_width * 3 * sizeof(float);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

    // register this buffer object with CUDA
    cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsWriteDiscard);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);


    glBindBuffer(GL_ARRAY_BUFFER, 0);

    runCuda(&cuda_vbo_resource, 0.0f);

    // You can unbind the VAO afterwards so other VAO calls won't accidentally modify this VAO, but this rarely happens. Modifying other
    // VAOs requires a call to glBindVertexArray anyways so we generally don't unbind VAOs (nor VBOs) when it's not directly necessary.
    glBindVertexArray(0);

    float t1 = (float)glfwGetTime();
    float t0 = (float)glfwGetTime();

    float timer = 0.0f;

    // render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {
        t1 = (float)glfwGetTime();
        deltaTime = t1 - t0;
        t0 = t1;

        timer += deltaTime * 1.0f;

        // input
        // -----
        processInput(window);

        runCuda(&cuda_vbo_resource, timer);

        // render
        // ------
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glEnable(GL_PROGRAM_POINT_SIZE);
        // render the triangle
        circleShader.use();
        //glPointSize(100.0f);


        glm::mat4 trIdentity = glm::mat4(1.0f);
        circleShader.setMat4("transform", camera.GetTransformMatrix());
        circleShader.setFloat("pointRadius", 25);
        circleShader.setFloat("pointScale", camera.GetZoom());
        circleShader.setVec3("Color", glm::vec3(1.0f, 0.0f, 0.0f));

        glBindVertexArray(VAO);
        glDrawArrays(GL_POINTS, 0, mesh_width);

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // optional: de-allocate all resources once they've outlived their purpose:
    // ------------------------------------------------------------------------
    glDeleteVertexArrays(1, &VAO);
    deleteVBO(&vbo, cuda_vbo_resource);

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
    return;
}

void deleteVBO(GLuint* vbo, struct cudaGraphicsResource* vbo_res)
{

    // unregister this buffer object with CUDA
    cudaGraphicsUnregisterResource(vbo_res);

    glBindBuffer(1, *vbo);
    glDeleteBuffers(1, vbo);

    *vbo = 0;
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.ProcessKeyboard(UP, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.ProcessKeyboard(DOWN, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.ProcessKeyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.ProcessKeyboard(RIGHT, deltaTime);
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
    {
        //cout << "Buton pressed in  " << deltaTime << endl;
        camera.SetDrag(true);
    }

    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE)
    {
        //cout << "Buton Released in  " << deltaTime << endl;
        camera.SetDrag(false);
    }
}


// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    float posX = 2 * (xpos - window_width / 2) / window_width;
    float posY = 2 * (window_height / 2 - ypos) / window_height;
    camera.SetCurrentPos(posX, posY);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    cout << "Scroll value  " << yoffset << endl;
    camera.ProcessMouseScroll(yoffset);
}
