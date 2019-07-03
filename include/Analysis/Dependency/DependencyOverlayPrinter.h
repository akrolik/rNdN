#pragma once

#include <string>
#include <unordered_map>

#include "Analysis/Dependency/DependencyOverlayConstVisitor.h"

namespace Analysis {

class DependencyOverlayPrinter : public DependencyOverlayConstVisitor
{
public:
	static std::string PrettyString(const DependencyOverlay *overlay);

	void Visit(const DependencyOverlay *overlay);

	void Visit(const CompoundDependencyOverlay<HorseIR::Function> *overlay);
	void Visit(const CompoundDependencyOverlay<HorseIR::IfStatement> *overlay);
	void Visit(const CompoundDependencyOverlay<HorseIR::WhileStatement> *overlay);
	void Visit(const CompoundDependencyOverlay<HorseIR::RepeatStatement> *overlay);

private:
	void Indent();
	unsigned int m_indent = 0;
	std::stringstream m_string;

	unsigned int m_nameIndex = 0;
	std::unordered_map<const HorseIR::Statement *, std::string> m_nameMap;
};

}
