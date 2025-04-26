#pragma once

#include <iostream>
#include <cstdio>
#include <glm/glm.hpp>

#define CUDA_CHECK(call) \
    do \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) \
        { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
			std::exit(EXIT_FAILURE); \
        } \
    } while (0)

template <typename T>
class CudaBuffer
{
public:
	CudaBuffer() = default;
	explicit CudaBuffer(size_t size) : m_size(size)
	{
		if (size > 0)
		{
			CUDA_CHECK(cudaMalloc(&m_data, size * sizeof(T)));
		}
	}

	~CudaBuffer() { Release(); }

	CudaBuffer(const CudaBuffer&) = default;
	CudaBuffer& operator=(const CudaBuffer&) = delete;

	CudaBuffer(CudaBuffer&& other) noexcept
		: m_data(other.m_data), m_size(other.m_size)
	{
		other.m_data = nullptr;
		other.m_size = 0;
	}

	CudaBuffer& operator=(CudaBuffer&& other) noexcept
	{
		if (this != &other)
		{
			Release();
			m_data = other.m_data;
			m_size = other.m_size;
			other.m_data = nullptr;
			other.m_size = 0;
		}
		return *this;
	}

	void Release()
	{
		if (m_data)
		{
			CUDA_CHECK(cudaDeviceSynchronize());
			CUDA_CHECK(cudaFree(m_data));
			m_data = nullptr;
		}
		m_size = 0;
	}

	void CopyFromHost(const T* hostData, size_t size)
	{
		if (m_data && size <= m_size)
		{
			CUDA_CHECK(cudaMemcpy(m_data, hostData, size * sizeof(T), cudaMemcpyHostToDevice));
		}
	}

	void CopyToHost(T* hostData, size_t size)
	{
		if (m_data && size <= m_size)
		{
			CUDA_CHECK(cudaMemcpy(hostData, m_data, size * sizeof(T), cudaMemcpyDeviceToHost));
		}
	}

	__device__ __host__ T* GetData()
	{
		if (!m_data)
		{
			printf("Warning! Trying to access null device pointer!\n");
		}

		return m_data;
	}
	__device__ __host__ [[nodiscard]] const T* GetData() const
	{
		if (!m_data)
		{
			printf("Warning! Trying to access null device pointer!\n");
		}

		return m_data;
	}

	void SetSize(const size_t size) { m_size = size; }

private:
	T* m_data = nullptr;
	size_t m_size = 0;
};