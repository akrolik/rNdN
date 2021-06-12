#pragma once

#include <string_view>

#include "Libraries/robin_hood.h"

#include "CUDA/Buffer.h"
#include "CUDA/Vector.h"

namespace Runtime {

class StringBucket
{
public:
	StringBucket(StringBucket const&) = delete;
	void operator=(StringBucket const&) = delete;

	static std::uint64_t HashString(const std::string_view& string);
	static const char *RecoverString(std::uint64_t index);

	static const CUDA::MappedVector<char>& GetBucket();
	static CUDA::Buffer *GetCache();

private:
	StringBucket() {}

	static StringBucket& GetInstance()
	{
		static StringBucket instance;
		return instance;
	}

	std::uint64_t m_index = 0;

	CUDA::MappedVector<char> m_bucket;
	robin_hood::unordered_map<std::uint64_t, std::uint64_t> m_hashMap;

	CUDA::Buffer *m_cacheBuffer = nullptr;
	std::size_t m_cacheSize = 0;
};

}
