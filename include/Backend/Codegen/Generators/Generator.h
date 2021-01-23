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

	[[noreturn]] void Error(const std::string& message)
	{
		Utils::Logger::LogError(Name() + ": Unable to generate " + message);
	}

	virtual std::string Name() const = 0;

protected:
	Builder &m_builder;
};

}
}
