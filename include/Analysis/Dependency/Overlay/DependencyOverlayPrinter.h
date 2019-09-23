#pragma once

#include <string>
#include <unordered_map>

#include "Analysis/Dependency/Overlay/DependencyOverlayConstVisitor.h"

#include "Analysis/Dependency/Overlay/DependencyOverlay.h"

#include "HorseIR/Tree/Tree.h"

namespace Analysis {

class DependencyOverlayPrinter : public DependencyOverlayConstVisitor
{
public:
	static std::string PrettyString(const DependencyOverlay *overlay);

	void Visit(const DependencyOverlay *overlay) override;

	void Visit(const FunctionDependencyOverlay *overlay) override;
	void Visit(const IfDependencyOverlay *overlay) override;
	void Visit(const WhileDependencyOverlay *overlay) override;
	void Visit(const RepeatDependencyOverlay *overlay) override;

private:
	void Indent();
	unsigned int m_indent = 0;
	std::stringstream m_string;

	unsigned int m_nameIndex = 0;
	std::unordered_map<const HorseIR::Statement *, std::string> m_nameMap;
};

}
