﻿
#include <glad/glad.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_gl_interop.h>

#include <GLFW/glfw3.h>

#include <glm/glm.hpp>

#include "../graphics/shaders/shader.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

using namespace std;


// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//
// PASO 1 - DECLARACION DE CONSTANTES, COMO DIMENSIONES , VBO Y VALORES DE CAMARA
//
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

const unsigned int window_width = 800;
const unsigned int window_height = 800;
const unsigned int mesh_width = 256;

// vbo variables
GLuint vbo;
struct cudaGraphicsResource* cuda_vbo_resource;
void* d_vbo_buffer = NULL;

void runKernel(float3* pos, unsigned int mesh_width, float time);
void runTest();
void runCuda(struct cudaGraphicsResource** vbo_resource, float time); 
void runDisplay();
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window, bool* fill);
void deleteVBO(GLuint* vbo, struct cudaGraphicsResource* vbo_res);
void circleShape(unsigned int*& indices, int verticesLength);

__global__ void simpleVBOKernel(float3 *pos, unsigned int width, float time)
{
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;

    // calculate uv coordinates
    float u = x / (float)width;


    // calculate simple sine wave pattern
    float freq = 1.0f; 
    float pi = 3.141592654f;
    float dTheta = (2.0f * pi) / (float)(width - 1);

    float pon = sinf(time) + 1;
    float w = (1 + cosf(u*12.0f*pi * pon)) *0.2f;
    float rad = 0.5f + w ;
    float finalX = rad * sinf(x * dTheta);
    float finalY = rad * cosf(x * dTheta);

    pos[x] = (x != 0) ? make_float3(finalX, finalY, 1.0f) : make_float3(0.0f, 0.0f, 1.0f);

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
    // dim3 block(8, 8, 1);
    // dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
    simpleVBOKernel << < 32, 8 >> > (pos, mesh_width, time);
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
    GLFWwindow* window = glfwCreateWindow(window_width, window_height, "Simple Cuda interop", NULL, NULL);
    if (window == NULL)
    {
        cout << "Failed to create GLFW window" << endl;
        glfwTerminate();
        return;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        cout << "Failed to initialize GLAD" << endl;
        return;
    }

    // build and compile our shader program
    // ------------------------------------
    Shader basicShader("../graphics/shaders/positionShader.vs", "../graphics/shaders/positionShader.fs"); // you can name your shader files however you like

    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    
    float* vertices;

    unsigned int* indices;

    circleShape(indices, mesh_width);

    unsigned int VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &EBO);
    // bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
    glBindVertexArray(VAO);

    // create buffer object
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    // initialize buffer object
    unsigned int size = mesh_width * 3 * sizeof(float);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

    // register this buffer object with CUDA
    cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsWriteDiscard);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, (mesh_width -1) * 3 * sizeof(unsigned int), indices, GL_STATIC_DRAW);


    

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);


    glBindBuffer(GL_ARRAY_BUFFER, 0);

    runCuda(&cuda_vbo_resource, 0.0f);




    // You can unbind the VAO afterwards so other VAO calls won't accidentally modify this VAO, but this rarely happens. Modifying other
    // VAOs requires a call to glBindVertexArray anyways so we generally don't unbind VAOs (nor VBOs) when it's not directly necessary.
    glBindVertexArray(0);

    bool fillPolygon = true;

    float t1 = (float)glfwGetTime();
    float t0 = (float)glfwGetTime();
    float delta = 0.0f;

    float timer = 0.0f;

    // render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {
        t1 = (float)glfwGetTime();
        delta = t1 - t0;
        t0 = t1;

        timer += delta * 1.0f;

        // input
        // -----
        processInput(window, &fillPolygon);

        runCuda(&cuda_vbo_resource, timer);

        // render
        // ------
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        if (fillPolygon)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        else
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);


        // render the triangle
        basicShader.use();
        glBindVertexArray(VAO);
        // glDrawArrays(GL_TRIANGLE_FAN, 0, mesh_width);
        glDrawElements(GL_TRIANGLES, (mesh_width - 1) * 3, GL_UNSIGNED_INT, 0);
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

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow* window, bool* fill)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
        *fill = false;

    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_RELEASE)
        *fill = true;
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

void deleteVBO(GLuint* vbo, struct cudaGraphicsResource* vbo_res)
{

    // unregister this buffer object with CUDA
    cudaGraphicsUnregisterResource(vbo_res);

    glBindBuffer(1, *vbo);
    glDeleteBuffers(1, vbo);

    *vbo = 0;
}

// Create indices for the mesh 
void circleShape(unsigned int*& indices, int verticesLength) {

    int indexSize = 3;
    indices = new unsigned int[(verticesLength - 1) * indexSize];

    for (unsigned int i = 0; i < verticesLength - 1; i++)
    {
        if (i < verticesLength - 2)
        {
            unsigned int temp1[] = {
                0, i + 1, i + 2 };
            copy(temp1, temp1 + (1 * indexSize), (indices + i * indexSize));
        }
        else {
            unsigned int temp1[] = {
                0, i + 1, 1 };
            copy(temp1, temp1 + (1 * indexSize), (indices + i * indexSize));
        }
    }


    cout << "Indices: " << endl;

    int li = 0;
    for (int i = 0; i < (verticesLength - 1)*3; i++) {
        cout << indices[i] << endl;
        li++;
    }
    cout << "length" << li << endl;
}