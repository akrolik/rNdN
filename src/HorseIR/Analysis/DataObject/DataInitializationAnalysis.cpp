#include "HorseIR/Analysis/DataObject/DataInitializationAnalysis.h"

#include "HorseIR/Utils/TypeUtils.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace HorseIR {
namespace Analysis {

void DataInitializationAnalysis::Analyze(const Function *function)
{
	auto& functionName = function->GetName();

	auto timeDataInitialization_start = Utils::Chrono::Start(Name + " '" + functionName + "'");
	function->Accept(*this);
	Utils::Chrono::End(timeDataInitialization_start);

	if (Utils::Options::IsFrontend_PrintAnalysis(ShortName, functionName))
	{
		Utils::Logger::LogInfo(Name + " '" + functionName + "'");
		Utils::Logger::LogInfo(DebugString(), 0, true, Utils::Logger::NoPrefix);
	}
}

bool DataInitializationAnalysis::VisitIn(const AssignStatement *assignS)
{
	// Traverse the RHS of the assignment and collect all initialized data

	assignS->GetExpression()->Accept(*this);
	return false;
}

const DataObject *DataInitializationAnalysis::GetDataObject(const CallExpression *call) const
{
	const auto& outputObjects = m_objectAnalysis.GetDataObjects(call);
	if (outputObjects.size() != 1)
	{
		Utils::Logger::LogError("Unable to initialize @" + call->GetFunctionLiteral()->GetFunction()->GetName() + " call, expected single output object, received " + std::to_string(outputObjects.size()));
	}
	return outputObjects.at(0);
}

bool DataInitializationAnalysis::VisitIn(const CallExpression *call)
{
	// Analyze the function call for initializations to the target

	const auto function = call->GetFunctionLiteral()->GetFunction();
	if (function->GetKind() == FunctionDeclaration::Kind::Builtin)
	{
		const auto builtinFunction = static_cast<const BuiltinFunction *>(function);
		auto primitive = builtinFunction->GetPrimitive();
		
		// Decompose the @each list functions, as the effect is the same as the nest

		switch (primitive)
		{
			case BuiltinFunction::Primitive::Each:
			case BuiltinFunction::Primitive::EachRight:
			case BuiltinFunction::Primitive::EachLeft:
			case BuiltinFunction::Primitive::EachItem:
			{
				const auto type = call->GetArgument(0)->GetType();
				const auto nestedFunction = TypeUtils::GetType<FunctionType>(type)->GetFunctionDeclaration();
				if (nestedFunction->GetKind() == FunctionDeclaration::Kind::Builtin)
				{
					const auto nestedBuiltin = static_cast<const BuiltinFunction *>(nestedFunction);
					primitive = nestedBuiltin->GetPrimitive();
				}
				break;
			}
		}

		switch (primitive)
		{
			case BuiltinFunction::Primitive::Length:
			case BuiltinFunction::Primitive::Sum:
			case BuiltinFunction::Primitive::Average:
			{
				auto outputObject = GetDataObject(call);
				m_dataInit.insert({outputObject, Initialization::Clear});
				break;
			}
			case BuiltinFunction::Primitive::Minimum:
			{
				auto outputObject = GetDataObject(call);
				m_dataInit.insert({outputObject, Initialization::Maximum});
				break;
			}
			case BuiltinFunction::Primitive::Maximum:
			{
				auto outputObject = GetDataObject(call);
				m_dataInit.insert({outputObject, Initialization::Minimum});
				break;
			}
			case BuiltinFunction::Primitive::IndexAssignment:
			{
				// For the index assignment function, we initialize the output with the input data
				// to handle data which is not set by the indexes

				auto inputObject = m_objectAnalysis.GetDataObject(call->GetArgument(0));
				auto outputObject = GetDataObject(call);
				m_dataCopies.insert({outputObject, inputObject});
				m_dataInit.insert({outputObject, Initialization::Copy});
				break;
			}
			case BuiltinFunction::Primitive::Raze:
			{
				// For list reductions, @raze is a passthrough

				auto inputObject = m_objectAnalysis.GetDataObject(call->GetArgument(0));
				if (m_dataInit.find(inputObject) != m_dataInit.end())
				{
					auto inputInit = m_dataInit.at(inputObject);
					auto outputObject = GetDataObject(call);
					m_dataInit.insert({outputObject, inputInit});
				}
				break;
			}
			case BuiltinFunction::Primitive::GPUHashJoinCreate:
			case BuiltinFunction::Primitive::GPUHashMemberCreate:
			{
				const auto& outputObjects = m_objectAnalysis.GetDataObjects(call);
				m_dataInit.insert({outputObjects.at(0), Initialization::Maximum});
				break;
			}
		}
	}
	return false;
}

std::string DataInitializationAnalysis::DebugString(unsigned int indent)
{
	std::string string(indent * Utils::Logger::IndentSize, ' ');

	auto first = true;
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
}
