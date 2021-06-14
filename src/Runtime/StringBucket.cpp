#include "Runtime/StringBucket.h"

#include "CUDA/BufferManager.h"

namespace Runtime {

std::uint64_t StringBucket::HashString(const std::string_view& string)
{
	auto& instance = GetInstance();
	auto& hashMap = instance.m_hashMap;

	auto hash = std::hash<std::string_view>{}(string);
	auto find = hashMap.find(hash);
	if (find != hashMap.end())
	{
		return find->second;
	}

	auto& bucket = instance.m_bucket;
	auto index = instance.m_index;

	// Add new value

	auto size = string.size();
	auto cstring = string.data();

	for (auto j = 0u; j < size; ++j)
	{
		bucket.push_back(cstring[j]);
	}

	auto paddedSize = size + 1;
	auto paddedFactor = 16;

	if (paddedSize % paddedFactor != 0)
	{
		paddedSize = paddedSize + (paddedFactor - paddedSize % paddedFactor);
	}

	for (auto j = size; j < paddedSize; ++j)
	{
		bucket.push_back('\0');
	}

	hashMap[hash] = index;
	instance.m_index += paddedSize;

	return index;
}

const char *StringBucket::RecoverString(std::uint64_t index)
{
	return GetInstance().m_bucket.data() + index;
}

const CUDA::MappedVector<char>& StringBucket::GetBucket()
{
	return GetInstance().m_bucket;
}

CUDA::Buffer *StringBucket::GetCache()
{
	auto& instance = GetInstance();
	if (instance.m_cacheBuffer == nullptr || instance.m_cacheSize != instance.m_bucket.size())
	{
		auto cacheSize = instance.m_bucket.size();
		auto cacheBuffer = CUDA::BufferManager::CreateBuffer(cacheSize);
		cacheBuffer->SetTag("like_cache");
		cacheBuffer->AllocateOnGPU();
		cacheBuffer->Clear(instance.m_cacheSize);

		if (instance.m_cacheBuffer != nullptr)
		{
			CUDA::Buffer::Copy(cacheBuffer, instance.m_cacheBuffer, instance.m_cacheSize);
			delete instance.m_cacheBuffer;
		}

		instance.m_cacheSize = cacheSize;
		instance.m_cacheBuffer = cacheBuffer;
	}
	return instance.m_cacheBuffer;
}

}
