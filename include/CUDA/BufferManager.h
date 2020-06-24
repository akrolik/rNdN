#pragma once

#include <vector>

#include "CUDA/Buffer.h"
#include "CUDA/ConstantBuffer.h"

namespace CUDA {

class BufferManager
{
public:
	// Manager lifecycle

	static void Initialize();
	static void Clear();
	static void Destroy();

	// Buffer

	static Buffer *CreateBuffer(size_t size);
	static ConstantBuffer *CreateConstantBuffer(size_t size);

protected:
	static BufferManager& GetInstance()
	{
		static BufferManager instance;
		return instance;
	}

	Buffer *GetPageBuffer(size_t size);

	std::vector<Buffer *> m_gpuBuffers;

	size_t m_page = 0;
	size_t m_sbrk = 0;
};

}
