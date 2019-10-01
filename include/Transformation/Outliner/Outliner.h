#pragma once

#include <vector>

#include "HorseIR/Tree/Tree.h"

namespace Transformation {

class Outliner : public HorseIR::ConstHierarchicalVisitor
{
public:
	// Transformation input/output

	void Outline(const HorseIR::Program *program);
	HorseIR::Program *GetOutlinedProgram() const { return m_outlinedProgram; }

	// Visitors

	bool VisitIn(const HorseIR::Program *program) override;
	bool VisitIn(const HorseIR::Module *module) override;
	bool VisitIn(const HorseIR::LibraryModule *module) override;
	bool VisitIn(const HorseIR::ImportDirective *import) override;
	bool VisitIn(const HorseIR::GlobalDeclaration *global) override;
	bool VisitIn(const HorseIR::Function *function) override;

	void VisitOut(const HorseIR::Program *program) override;
	void VisitOut(const HorseIR::Module *module) override;
	void VisitOut(const HorseIR::LibraryModule *module) override;

private:
	const HorseIR::Program *m_currentProgram = nullptr;
	
	HorseIR::Program *m_outlinedProgram;
	std::vector<HorseIR::Module *> m_outlinedModules;
	std::vector<HorseIR::ModuleContent *> m_outlinedContents;
};

}
