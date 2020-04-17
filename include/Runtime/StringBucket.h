#pragma once

#include <string>
#include <unordered_map>

namespace Runtime {

class StringBucket
{
public:
	StringBucket(StringBucket const&) = delete;
	void operator=(StringBucket const&) = delete;

	static std::uint64_t HashString(const char *string)
	{
		return HashString(std::string(string));
	}

	static std::uint64_t HashString(const std::string& string)
	{
		auto hash = std::hash<std::string>{}(string);
		GetInstance().m_bucket[hash] = string;
		return hash;
	}

	static const std::string& RecoverString(std::uint64_t hash)
	{
		return GetInstance().m_bucket.at(hash);
	}

private:
	StringBucket() {}

	static StringBucket& GetInstance()
	{
		static StringBucket instance;
		return instance;
	}

	std::unordered_map<std::uint64_t, std::string> m_bucket;
};

}
