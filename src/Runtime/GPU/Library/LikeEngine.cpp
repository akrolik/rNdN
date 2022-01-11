#include "Runtime/GPU/Library/LikeEngine.h"

#include "Runtime/StringBucket.h"
#include "Runtime/DataBuffers/BufferUtils.h"

#include "CUDA/BufferManager.h"
#include "CUDA/ConstantMappedBuffer.h"
#include "CUDA/KernelInvocation.h"
#include "CUDA/Utils.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"

namespace Runtime {
namespace GPU {

TypedVectorBuffer<std::int8_t> *LikeEngine::Like(const std::vector<const DataBuffer *>& arguments, bool cached)
{
	auto& gpuManager = m_runtime.GetGPUManager();
	auto libr3d3 = gpuManager.GetLibrary();

	// Get indexes and pattern buffers

	auto stringBuffer = BufferUtils::GetVectorBuffer<std::uint64_t>(arguments.at(0));
	auto patternBuffer = BufferUtils::GetVectorBuffer<std::uint64_t>(arguments.at(1));

	if (patternBuffer->GetElementCount() != 1)
	{
		Utils::Logger::LogError("GPU like library expects a single pattern argument, received " + std::to_string(patternBuffer->GetElementCount()));
	}

	// Configure the runtime thread layout

	auto vectorSize = stringBuffer->GetElementCount();
	auto blockSize = gpuManager.GetCurrentDevice()->GetMaxThreadsDimension(0);
	if (blockSize > vectorSize)
	{
		blockSize = vectorSize;
	}
	auto blockCount = (vectorSize + blockSize - 1) / blockSize;

	// Like kernel
	//   - Match
	//   - String
	//   - Cache
	//   - Pattern
	//   - Size

	auto likeKernel = libr3d3->GetKernel("like");

	CUDA::KernelInvocation likeInvocation(likeKernel);
	likeInvocation.SetBlockShape(blockSize, 1, 1);
	likeInvocation.SetGridShape(blockCount, 1, 1);

	// Pattern buffer

	auto patternData = patternBuffer->GetCPUReadBuffer();
	auto patternString = StringBucket::RecoverString(patternData->GetValue(0));
	auto patternSize = strlen(patternString);

	auto patternDataBuffer = CUDA::BufferManager::CreateConstantBuffer(patternSize + 1);
	patternDataBuffer->SetCPUBuffer(patternString);
	patternDataBuffer->AllocateOnGPU();
	patternDataBuffer->TransferToGPU();

	// Return buffer

	auto returnBuffer = new TypedVectorBuffer<std::int8_t>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Boolean), vectorSize);
	returnBuffer->RequireGPUConsistent(true);
	returnBuffer->Clear(DataBuffer::ClearMode::Zero);

	// Size buffer

	CUDA::TypedConstant<std::uint32_t> sizeConstant(vectorSize);

	// Parameters

	likeInvocation.AddParameter(*returnBuffer->GetGPUWriteBuffer());
	likeInvocation.AddParameter(*stringBuffer->GetGPUReadBuffer());

	// String pad parameter

	const auto& bucket = StringBucket::GetBucket();

	if (cached)
	{
		auto timeCache_start = Utils::Chrono::Start("String pad cache");

		// Cache kernel:
		//   - Cache
		//   - String
		//   - Pad
		//   - Size

		auto cacheKernel = libr3d3->GetKernel("like_cache");

		CUDA::KernelInvocation cacheInvocation(cacheKernel);
		cacheInvocation.SetBlockShape(blockSize, 1, 1);
		cacheInvocation.SetGridShape(blockCount, 1, 1);

		// Cache pad buffers

		auto cacheBuffer = StringBucket::GetCache();
		auto padBuffer = new CUDA::ConstantMappedBuffer(bucket.data());

		// Parameter setup

		cacheInvocation.AddParameter(*cacheBuffer);
		cacheInvocation.AddParameter(*stringBuffer->GetGPUReadBuffer());
		cacheInvocation.AddParameter(*padBuffer);
		cacheInvocation.AddParameter(sizeConstant);

		// Launch kernel

		cacheInvocation.SetDynamicSharedMemorySize(0);
		cacheInvocation.Launch();

		likeInvocation.AddParameter(*cacheBuffer);
		
		CUDA::Synchronize();

		Utils::Chrono::End(timeCache_start);
	}
	else
	{
		auto padBuffer = CUDA::BufferManager::CreateConstantBuffer(bucket.size());
		padBuffer->SetCPUBuffer(bucket.data());
		padBuffer->AllocateOnGPU();
		padBuffer->TransferToGPU();

		likeInvocation.AddParameter(*padBuffer);
	}

	likeInvocation.AddParameter(*patternDataBuffer);
	likeInvocation.AddParameter(sizeConstant);

	// Launch

	likeInvocation.SetDynamicSharedMemorySize(0);
	likeInvocation.Launch();

	// Return result, synchronized for timings

	CUDA::Synchronize();

	return returnBuffer;
}

}
}
