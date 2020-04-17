#pragma once

#include <string>
#include <unordered_map>

namespace Runtime {

class StringBucket
{
public:
	StringBucket(StringBucket const&) = delete;
	void operator=(StringBucket const&) = delete;

	static std::uint64_t HashString(const std::string& string)
	{
		auto& instance = GetInstance();
		auto& hashMap = instance.m_hashMap;

		auto hash = std::hash<std::string>{}(string);
		auto find = hashMap.find(hash);
		if (find != hashMap.end())
		{
			return find->second;
		}

		auto& bucket = instance.m_bucket;
		auto index = instance.m_index++;

		bucket.push_back(string);
		hashMap[hash] = index;

		return index;
	}

	static const std::string& RecoverString(std::uint64_t index)
	{
		return GetInstance().m_bucket.at(index);
	}

private:
	StringBucket() {}

	static StringBucket& GetInstance()
	{
		static StringBucket instance;
		return instance;
	}

	std::uint64_t m_index = 0;

	std::vector<std::string> m_bucket;
	std::unordered_map<std::uint64_t, std::uint64_t> m_hashMap;
};

}
