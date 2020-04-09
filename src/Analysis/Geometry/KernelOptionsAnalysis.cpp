#include "Analysis/Geometry/KernelOptionsAnalysis.h"

#include "Analysis/Geometry/GeometryAnalysis.h"
#include "Analysis/Geometry/KernelGeometryAnalysis.h"
#include "Analysis/Shape/Shape.h"
#include "Analysis/Shape/ShapeUtils.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Math.h"

namespace Analysis {

void KernelOptionsAnalysis::Analyze(const HorseIR::Function *function)
{
	auto timeKernelOptions_start = Utils::Chrono::Start("Kernel options '" + function->GetName() + "'");

	GeometryAnalysis geometryAnalysis(m_shapeAnalysis);
	geometryAnalysis.Analyze(function);

	KernelGeometryAnalysis kernelAnalysis(geometryAnalysis);
	kernelAnalysis.Analyze(function);

	auto timeCreateOptions_start = Utils::Chrono::Start("Create options");

	const auto& dataAnalysis = m_shapeAnalysis.GetDataAnalysis();

	// Construct the input options

	m_inputOptions = new Codegen::InputOptions();
	m_inputOptions->ThreadGeometry = kernelAnalysis.GetOperatingGeometry();

	for (const auto parameter : function->GetParameters())
	{
		m_inputOptions->Parameters[parameter->GetSymbol()] = parameter;

		m_inputOptions->ParameterShapes[parameter] = m_shapeAnalysis.GetParameterShape(parameter);
		m_inputOptions->ParameterObjects[parameter] = dataAnalysis.GetParameterObject(parameter);

		m_inputOptions->ParameterObjectMap[dataAnalysis.GetParameterObject(parameter)] = parameter;
	}

	// Collect declaration shapes

	function->Accept(*this);

	// Use the write shapes as that's what's actually active!

	m_inputOptions->ReturnShapes = m_shapeAnalysis.GetReturnShapes();
	m_inputOptions->ReturnWriteShapes = m_shapeAnalysis.GetReturnWriteShapes();

	// Check if the blocks should be in-order

	m_inputOptions->InOrderBlocks = IsInOrder(function);

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

bool KernelOptionsAnalysis::IsInOrder(const HorseIR::Function *function) const
{
	if (const auto vectorGeometry = ShapeUtils::GetShape<VectorShape>(m_inputOptions->ThreadGeometry))
	{
		// Input load compression

		for (const auto parameter : function->GetParameters())
		{
			const auto shape = m_inputOptions->ParameterShapes[parameter];
			if (const auto vectorShape = ShapeUtils::GetShape<VectorShape>(shape))
			{
				if (ShapeUtils::IsCompressedSize(vectorShape->GetSize(), vectorGeometry->GetSize()))
				{
					return true;
				}
			}
		}

		// Output write compression

		for (const auto returnShape : m_inputOptions->ReturnShapes)
		{
			if (const auto vectorShape = ShapeUtils::GetShape<VectorShape>(returnShape))
			{
				if (ShapeUtils::IsCompressedSize(vectorShape->GetSize(), vectorGeometry->GetSize()))
				{
					return true;
				}
			}
		}
	}
	return false;
}

bool KernelOptionsAnalysis::VisitIn(const HorseIR::Parameter *parameter)
{
	// Do nothing

	return true;
}

bool KernelOptionsAnalysis::VisitIn(const HorseIR::VariableDeclaration *declaration)
{
	const auto& endSet = m_shapeAnalysis.GetEndSet();

	m_inputOptions->Declarations[declaration->GetSymbol()] = declaration;
	m_inputOptions->DeclarationShapes[declaration] = endSet.first.at(declaration->GetSymbol());

	return true;
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
