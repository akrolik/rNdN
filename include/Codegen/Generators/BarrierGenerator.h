#pragma once

#include "Codegen/Generators/Generator.h"

#include "Analysis/Shape/Shape.h"
#include "Analysis/Shape/ShapeUtils.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/GeometryGenerator.h"

#include "PTX/PTX.h"

#include "Utils/Logger.h"

namespace Codegen {

class BarrierGenerator : public Generator
{
public:
	using Generator::Generator;

	void Generate()
	{
		auto& inputOptions = this->m_builder.GetInputOptions();

		// Barrier instructions depend on the thread geometry and the number of threads still active

		if (Analysis::ShapeUtils::IsShape<Analysis::VectorShape>(inputOptions.ThreadGeometry))
		{
			// All threads in the group participate in the barrier

			this->m_builder.AddStatement(new PTX::BarrierInstruction(new PTX::UInt32Value(0), true));
		}
		else if (Analysis::ShapeUtils::IsShape<Analysis::ListShape>(inputOptions.ThreadGeometry))
		{
			// The special barrier instruction requires threads be in multiples of the warp size

			auto warpSize = this->m_builder.GetTargetOptions().WarpSize;
			auto& kernelOptions = this->m_builder.GetKernelOptions();
			kernelOptions.SetThreadMultiple(warpSize);
			
			// Since some threads may have exited (cells out of range), count the number of active threads

			GeometryGenerator geometryGenerator(this->m_builder);
			auto activeThreads = geometryGenerator.GenerateActiveThreads();
			this->m_builder.AddStatement(new PTX::BarrierInstruction(new PTX::UInt32Value(0), activeThreads, true, true));
		}
		else
		{
			Utils::Logger::LogError("Unable to generate barrier instruction for thread geometry " + Analysis::ShapeUtils::ShapeString(inputOptions.ThreadGeometry));
		}
	}
};

}
