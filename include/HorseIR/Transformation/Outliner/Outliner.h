#pragma once

#include <vector>

#include "HorseIR/Tree/Tree.h"

namespace HorseIR {
namespace Transformation {

class Outliner : public ConstHierarchicalVisitor
{
public:
	// Transformation input/output

	Program *Outline(const Program *program);

	// Visitors

	bool VisitIn(const Program *program) override;
	bool VisitIn(const Module *module) override;
	bool VisitIn(const LibraryModule *module) override;
	bool VisitIn(const ImportDirective *import) override;
	bool VisitIn(const GlobalDeclaration *global) override;
	bool VisitIn(const Function *function) override;

	void VisitOut(const Program *program) override;
	void VisitOut(const Module *module) override;
	void VisitOut(const LibraryModule *module) override;

private:
	const Program *m_currentProgram = nullptr;
	
	Program *m_outlinedProgram;
	std::vector<Module *> m_outlinedModules;
	std::vector<ModuleContent *> m_outlinedContents;
};

}
}
