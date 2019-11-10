#include "Analysis/DataObject/DataCopyAnalysis.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace Analysis {

void DataCopyAnalysis::Analyze(const HorseIR::Function *function)
{
	auto timeDataCopy_start = Utils::Chrono::Start();
	function->Accept(*this);
	auto timeDataCopy = Utils::Chrono::End(timeDataCopy_start);

	if (Utils::Options::Present(Utils::Options::Opt_Print_analysis))
	{
		Utils::Logger::LogInfo("Data copy analysis");

		Utils::Logger::LogInfo(DebugString(), 0, true, Utils::Logger::NoPrefix);
	}
	Utils::Logger::LogTiming("Data copy analysis", timeDataCopy);
}

bool DataCopyAnalysis::VisitIn(const HorseIR::AssignStatement *assignS)
{
	// Traverse the RHS of the assignment and collect all copy initialized data

	assignS->GetExpression()->Accept(*this);
	return false;
}

bool DataCopyAnalysis::VisitIn(const HorseIR::CallExpression *call)
{
	// Analyze the function call for copies to the target

	auto function = call->GetFunctionLiteral()->GetFunction();
	if (function->GetKind() == HorseIR::FunctionDeclaration::Kind::Builtin)
	{
		auto builtinFunction = static_cast<const HorseIR::BuiltinFunction *>(function);
		if (builtinFunction->GetPrimitive() == HorseIR::BuiltinFunction::Primitive::IndexAssignment)
		{
			// For the index assignment function, we initialize the output with the input data
			// to handle data which is not set by the indexes

			auto inputObject = m_objectAnalysis.GetDataObject(call->GetArgument(0));
			auto outputObjects = m_objectAnalysis.GetDataObjects(call);

			if (outputObjects.size() != 1)
			{
				Utils::Logger::LogError("Unable to create copy for @index_a call, expected single output object, received " + std::to_string(outputObjects.size()));
			}

			m_dataCopies.insert({outputObjects.at(0), inputObject});
		}
	}
	return false;
}

std::string DataCopyAnalysis::DebugString(unsigned int indent)
{
	std::string string;
	for (unsigned int i = 0; i < indent; ++i)
	{
		string += "\t";
	}

	bool first = true;
	for (const auto& copy : m_dataCopies)
	{
		if (!first)
		{
			string += ", ";
		}
		first = false;
		string += copy.second->ToString() + " -> " + copy.first->ToString();
	}
	return string;
}

}
