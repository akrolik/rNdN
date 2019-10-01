#include "Transformation/Outliner/Outliner.h"

#include "Analysis/Compatibility/CompatibilityAnalysis.h"
#include "Analysis/Compatibility/Geometry/GeometryAnalysis.h"
#include "Analysis/Dependency/DependencyAccessAnalysis.h"
#include "Analysis/Dependency/DependencyAnalysis.h"
#include "Analysis/Dependency/DependencySubgraphAnalysis.h"
#include "Analysis/Dependency/Overlay/DependencyOverlayPrinter.h"
#include "Analysis/Shape/ShapeAnalysis.h"

#include "HorseIR/Analysis/FlowAnalysisPrinter.h"

#include "Transformation/Outliner/OutlineBuilder.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace Transformation {

void Outliner::Outline(const HorseIR::Program *program)
{
	Utils::Logger::LogSection("Outlining program");

	m_currentProgram = program;
	m_outlinedProgram = nullptr;
	program->Accept(*this);

	if (Utils::Options::Present(Utils::Options::Opt_Print_outline))
	{
		Utils::Logger::LogInfo("Outlined HorseIR program");

		auto outlinedString = HorseIR::PrettyPrinter::PrettyString(m_outlinedProgram);
		Utils::Logger::LogInfo(outlinedString, 0, true, Utils::Logger::NoPrefix);
	}
}

bool Outliner::VisitIn(const HorseIR::Program *program)
{
	m_outlinedModules.clear();
	return true;
}

void Outliner::VisitOut(const HorseIR::Program *program)
{
	m_outlinedProgram = new HorseIR::Program(m_outlinedModules);
}

bool Outliner::VisitIn(const HorseIR::Module *module)
{
	m_outlinedContents.clear();
	return true;
}

bool Outliner::VisitIn(const HorseIR::LibraryModule *module)
{
	return false;
}

void Outliner::VisitOut(const HorseIR::LibraryModule *module)
{
	// Do nothing
}

void Outliner::VisitOut(const HorseIR::Module *module)
{
	auto outlinedModule = new HorseIR::Module(module->GetName(), m_outlinedContents);
	m_outlinedModules.push_back(outlinedModule);
}

bool Outliner::VisitIn(const HorseIR::ImportDirective *import)
{
	m_outlinedContents.push_back(import->Clone());
	return false;
}

bool Outliner::VisitIn(const HorseIR::GlobalDeclaration *global)
{
	m_outlinedContents.push_back(global->Clone());
	return false;
}

bool Outliner::VisitIn(const HorseIR::Function *function)
{
	// Outline GPU kernels
	// 1. Partition the overlay into an outlined overlay with nested functions
	// 2. Build the partitioned graph into functions collected in a vector

	// Dependency analysis (access and builder)

	auto timeDependencies_start = Utils::Chrono::Start();

	Analysis::DependencyAccessAnalysis accessAnalysis(m_currentProgram);
	accessAnalysis.Analyze(function);

	Analysis::DependencyAnalysis dependencyAnalysis(accessAnalysis);
	dependencyAnalysis.Build(function);

	auto dependencyOverlay = dependencyAnalysis.GetOverlay();

	Analysis::DependencySubgraphAnalysis dependencySubgraphAnalysis;
	dependencySubgraphAnalysis.Analyze(dependencyOverlay);

	auto timeDependencies = Utils::Chrono::End(timeDependencies_start);

	if (Utils::Options::Present(Utils::Options::Opt_Print_analysis))
	{
		Utils::Logger::LogInfo("Dependency access analysis");

		auto accessString = HorseIR::FlowAnalysisPrinter<Analysis::DependencyAccessProperties>::PrettyString(accessAnalysis, function);
		Utils::Logger::LogInfo(accessString, 0, true, Utils::Logger::NoPrefix);

		Utils::Logger::LogInfo("Dependency graph");

		auto dependencyString = Analysis::DependencyOverlayPrinter::PrettyString(dependencyOverlay);
		Utils::Logger::LogInfo(dependencyString, 0, true, Utils::Logger::NoPrefix);
	}

	Utils::Logger::LogTiming("Dependency analysis", timeDependencies);

	// Perform a conservative shape analysis

	auto timeShapes_start = Utils::Chrono::Start();

	Analysis::ShapeAnalysis shapeAnalysis(m_currentProgram);
	shapeAnalysis.Analyze(function);

	auto timeShapes = Utils::Chrono::End(timeShapes_start);

	if (Utils::Options::Present(Utils::Options::Opt_Print_analysis))
	{
		Utils::Logger::LogInfo("Shapes analysis");

		auto shapesString = HorseIR::FlowAnalysisPrinter<Analysis::ShapeAnalysisProperties>::PrettyString(shapeAnalysis, function);
		Utils::Logger::LogInfo(shapesString, 0, true, Utils::Logger::NoPrefix);
	}

	Utils::Logger::LogTiming("Shape analysis", timeShapes);

	// Compatibility analysis from dependency and geometry analyses

	auto timeCompatibility_start = Utils::Chrono::Start();

	Analysis::GeometryAnalysis geometryAnalysis(shapeAnalysis);
	geometryAnalysis.Analyze(function);

	Analysis::CompatibilityAnalysis compatibilityAnalysis(geometryAnalysis);
	compatibilityAnalysis.Analyze(dependencyOverlay);

	auto compatibilityOverlay = compatibilityAnalysis.GetOverlay();

	Analysis::DependencySubgraphAnalysis compatibilitySubgraphAnalysis;
	compatibilitySubgraphAnalysis.Analyze(compatibilityOverlay);

	auto timeCompatibility = Utils::Chrono::End(timeCompatibility_start);

	if (Utils::Options::Present(Utils::Options::Opt_Print_analysis))
	{
		Utils::Logger::LogInfo("Compatibility graph");

		auto compatibilityString = Analysis::DependencyOverlayPrinter::PrettyString(compatibilityOverlay);
		Utils::Logger::LogInfo(compatibilityString, 0, true, Utils::Logger::NoPrefix);
	}

	Utils::Logger::LogTiming("Compatibility analysis", timeCompatibility);

	auto timeBuilder_start = Utils::Chrono::Start();

	OutlineBuilder builder;
	builder.Build(compatibilityOverlay);

	auto timeBuilder = Utils::Chrono::End(timeBuilder_start);

	auto outlinedFunctions = builder.GetFunctions();
	m_outlinedContents.insert(std::end(m_outlinedContents), std::begin(outlinedFunctions), std::end(outlinedFunctions));

	Utils::Logger::LogTiming("Builder", timeBuilder);

	return false;
}

}
