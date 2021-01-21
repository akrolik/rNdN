#pragma once

#include "Backend/Codegen/Builder.h"

#include "Utils/Logger.h"

#include <string>

namespace Backend { 
namespace Codegen {

class Generator
{
public:
	Generator(Builder& builder) : m_builder(builder) {}

	[[noreturn]] void Error(const std::string& error)
	{
		//TODO: error message
		Utils::Logger::LogError("");
	}

	virtual std::string Name() const = 0;

protected:
	Builder &m_builder;
};

}
}
