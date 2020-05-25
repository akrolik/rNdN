#pragma once

#include "Codegen/Builder.h"

#include "HorseIR/Tree/Tree.h"

#include "Utils/Logger.h"

namespace Codegen {

class Generator
{
public:
	Generator(Builder& builder) : m_builder(builder) {}

	template<class G, typename... N>
	friend void DispatchType(G&, const HorseIR::Type*, N ...);

	template<class G, typename... N>
	friend void DispatchList(G&, const HorseIR::ListType*, N ...);

	template<template <typename> typename D, class G, typename... N>
	friend void Dispatch(G &generator, const HorseIR::BasicType *type, unsigned int i, N ...nodes);

	[[noreturn]] void Error(const std::string& message) const
	{
		Utils::Logger::LogError(m_builder.GetContextString(Name() + ": Unable to generate " + message + " [geometry = " + Analysis::ShapeUtils::ShapeString(this->m_builder.GetInputOptions().ThreadGeometry) + "]"));
	}

	virtual std::string Name() const = 0;

protected:
	Builder &m_builder;
};

}
