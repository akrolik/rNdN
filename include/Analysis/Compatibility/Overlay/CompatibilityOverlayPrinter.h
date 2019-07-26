#pragma once

#include <string>
#include <unordered_map>

#include "Analysis/Compatibility/Overlay/CompatibilityOverlayConstVisitor.h"

#include "Analysis/Compatibility/Overlay/CompatibilityOverlay.h"

#include "HorseIR/Tree/Tree.h"

namespace Analysis {

class CompatibilityOverlayPrinter : public CompatibilityOverlayConstVisitor
{
public:
	static std::string PrettyString(const CompatibilityOverlay *overlay);

	void Visit(const CompatibilityOverlay *overlay) override;

	void Visit(const FunctionCompatibilityOverlay *overlay) override;
	void Visit(const IfCompatibilityOverlay *overlay) override;
	void Visit(const WhileCompatibilityOverlay *overlay) override;
	void Visit(const RepeatCompatibilityOverlay *overlay) override;

private:
	void Indent();
	unsigned int m_indent = 0;
	std::stringstream m_string;

	unsigned int m_nameIndex = 0;
	std::unordered_map<const HorseIR::Statement *, std::string> m_nameMap;
};

}
