#include "Analysis/DataObject/DataInitializationAnalysis.h"

#include "HorseIR/Utils/TypeUtils.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace Analysis {

void DataInitializationAnalysis::Analyze(const HorseIR::Function *function)
{
	auto timeDataInitialization_start = Utils::Chrono::Start("Data initialization analysis");
	function->Accept(*this);
	Utils::Chrono::End(timeDataInitialization_start);

	if (Utils::Options::Present(Utils::Options::Opt_Print_analysis))
	{
		Utils::Logger::LogInfo("Data initialization analysis");
		Utils::Logger::LogInfo(DebugString(), 0, true, Utils::Logger::NoPrefix);
	}
}

bool DataInitializationAnalysis::VisitIn(const HorseIR::AssignStatement *assignS)
{
	// Traverse the RHS of the assignment and collect all initialized data

	assignS->GetExpression()->Accept(*this);
	return false;
}

const DataObject *DataInitializationAnalysis::GetDataObject(const HorseIR::CallExpression *call) const
{
	const auto& outputObjects = m_objectAnalysis.GetDataObjects(call);
	if (outputObjects.size() != 1)
	{
		Utils::Logger::LogError("Unable to initialize @" + call->GetFunctionLiteral()->GetFunction()->GetName() + " call, expected single output object, received " + std::to_string(outputObjects.size()));
	}
	return outputObjects.at(0);
}

bool DataInitializationAnalysis::VisitIn(const HorseIR::CallExpression *call)
{
	// Analyze the function call for initializations to the target

	const auto function = call->GetFunctionLiteral()->GetFunction();
	if (function->GetKind() == HorseIR::FunctionDeclaration::Kind::Builtin)
	{
		const auto builtinFunction = static_cast<const HorseIR::BuiltinFunction *>(function);
		auto primitive = builtinFunction->GetPrimitive();
		
		// Decompose the @each list functions, as the effect is the same as the nest

		switch (primitive)
		{
			case HorseIR::BuiltinFunction::Primitive::Each:
			case HorseIR::BuiltinFunction::Primitive::EachRight:
			case HorseIR::BuiltinFunction::Primitive::EachLeft:
			case HorseIR::BuiltinFunction::Primitive::EachItem:
			{
				const auto type = call->GetArgument(0)->GetType();
				const auto nestedFunction = HorseIR::TypeUtils::GetType<HorseIR::FunctionType>(type)->GetFunctionDeclaration();
				if (nestedFunction->GetKind() == HorseIR::FunctionDeclaration::Kind::Builtin)
				{
					const auto nestedBuiltin = static_cast<const HorseIR::BuiltinFunction *>(nestedFunction);
					primitive = nestedBuiltin->GetPrimitive();
				}
				break;
			}
		}

		switch (primitive)
		{
			case HorseIR::BuiltinFunction::Primitive::Length:
			case HorseIR::BuiltinFunction::Primitive::Sum:
			case HorseIR::BuiltinFunction::Primitive::Average:
			{
				auto outputObject = GetDataObject(call);
				m_dataInit.insert({outputObject, Initialization::Clear});
				break;
			}
			case HorseIR::BuiltinFunction::Primitive::Minimum:
			{
				auto outputObject = GetDataObject(call);
				m_dataInit.insert({outputObject, Initialization::Maximum});
				break;
			}
			case HorseIR::BuiltinFunction::Primitive::Maximum:
			{
				auto outputObject = GetDataObject(call);
				m_dataInit.insert({outputObject, Initialization::Minimum});
				break;
			}
			case HorseIR::BuiltinFunction::Primitive::IndexAssignment:
			{
				// For the index assignment function, we initialize the output with the input data
				// to handle data which is not set by the indexes

				auto inputObject = m_objectAnalysis.GetDataObject(call->GetArgument(0));
				auto outputObject = GetDataObject(call);
				m_dataCopies.insert({outputObject, inputObject});
				m_dataInit.insert({outputObject, Initialization::Copy});
				break;
			}
			case HorseIR::BuiltinFunction::Primitive::Raze:
			{
				// For list reductions, @raze is a passthrough

				auto inputObject = m_objectAnalysis.GetDataObject(call->GetArgument(0));
				if (m_dataInit.find(inputObject) != m_dataInit.end())
				{
					auto inputInit = m_dataInit.at(inputObject);
					auto outputObject = GetDataObject(call);
					m_dataInit.insert({outputObject, inputInit});
				}
			}
		}
	}
	return false;
}

std::string DataInitializationAnalysis::DebugString(unsigned int indent)
{
	std::string string;
	for (unsigned int i = 0; i < indent; ++i)
	{
		string += "\t";
	}

	bool first = true;
	for (const auto& initialization : m_dataInit)
	{
		if (!first)
		{
			string += ", ";
		}
		first = false;

		string += initialization.first->ToString() + " -> ";
		switch (initialization.second)
		{
			case Initialization::Clear:
			{
				string += "<clear>";
				break;
			}
			case Initialization::Minimum:
			{
				string += "<min>";
				break;
			}
			case Initialization::Maximum:
			{
				string += "<min>";
				break;
			}
			case Initialization::Copy:
			{
				string += "<copy> " + GetDataCopy(initialization.first)->ToString();
				break;
			}
		}
	}
	return string;
}

}
