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
	void Build(const Analysis::CompatibilityOverlay *overlay);
	const std::vector<HorseIR::Function *>& GetFunctions() const { return m_functions; }

	void Visit(const Analysis::CompatibilityOverlay *overlay) override;

	void Visit(const Analysis::FunctionCompatibilityOverlay *overlay) override;
	void Visit(const Analysis::IfCompatibilityOverlay *overlay) override;
	void Visit(const Analysis::WhileCompatibilityOverlay *overlay) override;
	void Visit(const Analysis::RepeatCompatibilityOverlay *overlay) override;

private:
	Analysis::CompatibilityOverlay *GetChildOverlay(const std::vector<Analysis::CompatibilityOverlay *>& childOverlays, const HorseIR::Statement *statement) const;
	unsigned int GetOutDegree(const Analysis::CompatibilityOverlay *overlay) const;

	std::vector<HorseIR::Function *> m_functions;

	unsigned int m_kernelIndex = 1;
	const HorseIR::Function *m_currentFunction = nullptr;
	std::stack<std::vector<HorseIR::Statement *>> m_statements;
};

}
