#pragma once

#include <stack>
#include <vector>

#include "Analysis/Compatibility/Overlay/CompatibilityOverlayConstVisitor.h"

#include "Analysis/Compatibility/Overlay/CompatibilityOverlay.h"

#include "HorseIR/Tree/Tree.h"

namespace Transformation {

class OutlineBuilder : public Analysis::CompatibilityOverlayConstVisitor
{
public:
	OutlineBuilder(std::vector<HorseIR::Function *>& functions) : m_functions(functions) {}

	void Build(const Analysis::CompatibilityOverlay *overlay);

	void Visit(const Analysis::CompatibilityOverlay *overlay) override;

	void Visit(const Analysis::FunctionCompatibilityOverlay *overlay) override;
	void Visit(const Analysis::IfCompatibilityOverlay *overlay) override;
	void Visit(const Analysis::WhileCompatibilityOverlay *overlay) override;
	void Visit(const Analysis::RepeatCompatibilityOverlay *overlay) override;

private:
	std::vector<HorseIR::Function *>& m_functions;

	std::stack<std::pair<const HorseIR::Function *, unsigned int>> m_containerFunctions;
	std::stack<std::vector<HorseIR::Statement *>> m_statements;
};

}
