#include "HorseIR/Transformation/Outliner/Outliner.h"

#include "HorseIR/Analysis/Compatibility/CompatibilityAnalysis.h"
#include "HorseIR/Analysis/Dependency/DependencyAccessAnalysis.h"
#include "HorseIR/Analysis/Dependency/DependencyAnalysis.h"
#include "HorseIR/Analysis/Dependency/DependencySubgraphAnalysis.h"
#include "HorseIR/Analysis/DataObject/DataObjectAnalysis.h"
#include "HorseIR/Analysis/Geometry/GeometryAnalysis.h"
#include "HorseIR/Analysis/Shape/ShapeAnalysis.h"

#include "HorseIR/Semantics/SemanticAnalysis.h"

#include "HorseIR/Transformation/Outliner/OutlineBuilder.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace HorseIR {
namespace Transformation {

Program *Outliner::Outline(const Program *program)
{
	auto timeOutliner_start = Utils::Chrono::Start("Outliner");

	m_currentProgram = program;
	m_outlinedProgram = nullptr;
	program->Accept(*this);

	if (Utils::Options::IsFrontend_PrintOutline())
	{
		Utils::Logger::LogInfo("Outlined HorseIR program");

		auto outlinedString = PrettyPrinter::PrettyString(m_outlinedProgram);
		Utils::Logger::LogInfo(outlinedString, 0, true, Utils::Logger::NoPrefix);
	}

	// Re-run the semantic analysis to build the AST symbol table links
	
	SemanticAnalysis::Analyze(m_outlinedProgram);

	Utils::Chrono::End(timeOutliner_start);

	return m_outlinedProgram;
}

bool Outliner::VisitIn(const Program *program)
{
	m_outlinedModules.clear();
	return true;
}

void Outliner::VisitOut(const Program *program)
{
	m_outlinedProgram = new Program(m_outlinedModules);
}

bool Outliner::VisitIn(const Module *module)
{
	m_outlinedContents.clear();
	return true;
}

bool Outliner::VisitIn(const LibraryModule *module)
{
	return false;
}

void Outliner::VisitOut(const LibraryModule *module)
{
	// Do nothing
}

void Outliner::VisitOut(const Module *module)
{
	auto outlinedModule = new Module(module->GetName(), m_outlinedContents);
	m_outlinedModules.push_back(outlinedModule);
}

bool Outliner::VisitIn(const ImportDirective *import)
{
	m_outlinedContents.push_back(import->Clone());
	return false;
}

bool Outliner::VisitIn(const GlobalDeclaration *global)
{
	m_outlinedContents.push_back(global->Clone());
	return false;
}

bool Outliner::VisitIn(const Function *function)
{
	// Outline GPU kernels
	// 1. Partition the overlay into an outlined overlay with nested functions
	// 2. Build the partitioned graph into functions collected in a vector

	// Dependency analysis (access and builder)

	auto timeOutline_start = Utils::Chrono::Start("Outline function '" + function->GetName() + "'");

	Analysis::DependencyAccessAnalysis accessAnalysis(m_currentProgram);
	accessAnalysis.Analyze(function);

	Analysis::DependencyAnalysis dependencyAnalysis(accessAnalysis);
	dependencyAnalysis.Build(function);

	auto dependencyOverlay = dependencyAnalysis.GetOverlay();

	Analysis::DependencySubgraphAnalysis dependencySubgraphAnalysis;
	dependencySubgraphAnalysis.Analyze(dependencyOverlay);

	// Perform a conservative shape analysis

	Analysis::DataObjectAnalysis dataAnalysis(m_currentProgram);
	dataAnalysis.Analyze(function);

	Analysis::ShapeAnalysis shapeAnalysis(dataAnalysis, m_currentProgram);
	shapeAnalysis.Analyze(function);

	// Compatibility analysis from dependency and geometry analyses

	Analysis::GeometryAnalysis geometryAnalysis(shapeAnalysis);
	geometryAnalysis.Analyze(function);

	Analysis::CompatibilityAnalysis compatibilityAnalysis(geometryAnalysis);
	compatibilityAnalysis.Analyze(dependencyOverlay);

	auto compatibilityOverlay = compatibilityAnalysis.GetOverlay();

	Analysis::DependencySubgraphAnalysis compatibilitySubgraphAnalysis;
	compatibilitySubgraphAnalysis.Analyze(compatibilityOverlay);

	OutlineBuilder builder;
	auto outlinedFunctions = builder.Build(compatibilityOverlay);

	m_outlinedContents.insert(std::begin(m_outlinedContents), new ImportDirective("GPU", "*"));
	m_outlinedContents.insert(std::end(m_outlinedContents), std::begin(outlinedFunctions), std::end(outlinedFunctions));

	Utils::Chrono::End(timeOutline_start);

	return false;
}

}
}
