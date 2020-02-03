#include "Analysis/Geometry/KernelOptionsAnalysis.h"

#include "Analysis/Geometry/GeometryAnalysis.h"
#include "Analysis/Geometry/KernelAnalysis.h"
#include "Analysis/Shape/Shape.h"
#include "Analysis/Shape/ShapeUtils.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Math.h"

namespace Analysis {

void KernelOptionsAnalysis::Analyze(const HorseIR::Function *function, const ShapeAnalysis& shapeAnalysis)
{
	auto timeKernelOptions_start = Utils::Chrono::Start("Kernel options '" + function->GetName() + "'");

	GeometryAnalysis geometryAnalysis(shapeAnalysis);
	geometryAnalysis.Analyze(function);

	KernelAnalysis kernelAnalysis(geometryAnalysis);
	kernelAnalysis.Analyze(function);

	auto timeCreateOptions_start = Utils::Chrono::Start("Create options");

	const auto& dataAnalysis = shapeAnalysis.GetDataAnalysis();

	// Construct the input options

	//TODO: Determine order
	m_inputOptions = new Codegen::InputOptions();
	m_inputOptions->ThreadGeometry = kernelAnalysis.GetOperatingGeometry();
	m_inputOptions->InOrderBlocks = ShapeUtils::IsShape<VectorShape>(m_inputOptions->ThreadGeometry);

	for (const auto& parameter : function->GetParameters())
	{
		m_inputOptions->Parameters[parameter->GetSymbol()] = parameter;

		m_inputOptions->ParameterShapes[parameter] = shapeAnalysis.GetParameterShape(parameter);
		m_inputOptions->ParameterObjects[parameter] = dataAnalysis.GetParameterObject(parameter);

		m_inputOptions->ParameterObjectMap[dataAnalysis.GetParameterObject(parameter)] = parameter;
	}

	// Use the write shapes as that's what's actually active!

	m_inputOptions->ReturnShapes = shapeAnalysis.GetReturnShapes();
	m_inputOptions->ReturnWriteShapes = shapeAnalysis.GetReturnWriteShapes();

	// Specify the number of threads for each cell computation in list thread geometry

	if (const auto listShape = Analysis::ShapeUtils::GetShape<Analysis::ListShape>(m_inputOptions->ThreadGeometry))
	{
		if (Analysis::ShapeUtils::IsDynamicShape(listShape))
		{
			m_inputOptions->ListCellThreads = Codegen::InputOptions::DynamicSize;
		}
		else
		{
			m_inputOptions->ListCellThreads = GetAverageCellSize(listShape);
		}
	}

	if (Utils::Options::Get<>(Utils::Options::Opt_Print_analysis))
	{
		Utils::Logger::LogInfo("Input Options: " + function->GetName());
		Utils::Logger::LogInfo(m_inputOptions->ToString(), 1);
	}

	Utils::Chrono::End(timeCreateOptions_start);
	Utils::Chrono::End(timeKernelOptions_start);
}

std::uint32_t KernelOptionsAnalysis::GetAverageCellSize(const Analysis::ListShape *shape) const
{
	// Form a vector of cell sizes, for lists of constan-sized vectors

	std::vector<std::uint32_t> cellSizes;
	for (const auto cellShape : shape->GetElementShapes())
	{
		if (const auto vectorShape = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(cellShape))
		{
			if (const auto constantSize = Analysis::ShapeUtils::GetSize<Analysis::Shape::ConstantSize>(vectorShape->GetSize()))
			{
				cellSizes.push_back(constantSize->GetValue());
				continue;
			}
		}
		Utils::Logger::LogError("Unable to get constant cell sizes for list shape " + Analysis::ShapeUtils::ShapeString(shape));
	}

	const auto averageCellSize = std::accumulate(cellSizes.begin(), cellSizes.end(), 0) / cellSizes.size();
	return Utils::Math::Power2(averageCellSize);
}

}
