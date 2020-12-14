#include "HorseIR/Analysis/DataObject/DataObjectAnalysis.h"

#include "HorseIR/Utils/TypeUtils.h"

#include "Utils/Logger.h"

namespace HorseIR {
namespace Analysis {

void DataObjectAnalysis::Visit(const Parameter *parameter)
{
	// Initialize all incoming parameters with unique data objects

	m_currentOutSet = m_currentInSet;
	
	auto symbol = parameter->GetSymbol();
	if (m_currentOutSet.find(symbol) == m_currentOutSet.end())
	{
		m_currentOutSet[parameter->GetSymbol()] = new DataObject();
	}

	m_parameterObjects[parameter] = m_currentOutSet.at(parameter->GetSymbol());
}

void DataObjectAnalysis::Visit(const AssignStatement *assignS)
{
	// For each target of the assignment, kill the previous data objects (if any)
	// and add a new value to the map associating the target to the data object

	m_currentOutSet = m_currentInSet;

	auto expression = assignS->GetExpression();
	expression->Accept(*this);
	auto dataObjects = m_expressionObjects.at(expression);

	auto index = 0u;
	for (const auto target : assignS->GetTargets())
	{
		// Construct the new value for the set

		auto symbol = target->GetSymbol();
		m_currentOutSet[symbol] = dataObjects.at(index++);
	}
}

void DataObjectAnalysis::Visit(const BlockStatement *blockS)
{
	// Visit all statements within the block and compute the sets

	ForwardAnalysis<DataObjectProperties>::Visit(blockS);

	// Kill all symbols that were part of the block

	auto symbolTable = blockS->GetSymbolTable();
	auto it = m_currentOutSet.begin();
	while (it != m_currentOutSet.end())
	{
		auto symbol = it->first;
		if (symbolTable->ContainsSymbol(symbol))
		{
			it = m_currentOutSet.erase(it);
		}
		else
		{
			++it;
		}
	}
}

void DataObjectAnalysis::Visit(const ReturnStatement *returnS)
{
	ForwardAnalysis<DataObjectProperties>::Visit(returnS);

	std::vector<const DataObject *> returnObjects;
	for (const auto& operand : returnS->GetOperands())
	{
		returnObjects.push_back(GetDataObject(operand));
	}

	if (m_returnObjects.size() == 0)
	{
		m_returnObjects = returnObjects;
	}
	else
	{
		for (auto i = 0u; i < m_returnObjects.size(); ++i)
		{
			auto oldObject = m_returnObjects.at(i);
			auto newObject = returnObjects.at(i);

			if (*oldObject != *newObject)
			{
				m_returnObjects.at(i) = new DataObject();
			}
		}
	}
}

void DataObjectAnalysis::Visit(const CastExpression *cast)
{
	// Create new data objects for other (non-identifier) expressions on RHS of assignment

	m_expressionObjects[cast] = {new DataObject()};
}

void DataObjectAnalysis::Visit(const CallExpression *call)
{
	// Accumulate data objects for arguments

	std::vector<const DataObject *> argumentObjects;
	for (const auto argument : call->GetArguments())
	{
		argument->Accept(*this);
		argumentObjects.push_back(GetDataObject(argument));
	}
	m_expressionObjects[call] = AnalyzeCall(call->GetFunctionLiteral()->GetFunction(), call->GetArguments(), argumentObjects);
}

std::vector<const DataObject *> DataObjectAnalysis::AnalyzeCall(const FunctionDeclaration *function, const std::vector<const Operand *>& arguments, const std::vector<const DataObject *>& argumentObjects)
{
	switch (function->GetKind())
	{
		case FunctionDeclaration::Kind::Builtin:
			return AnalyzeCall(static_cast<const BuiltinFunction *>(function), arguments, argumentObjects);
		case FunctionDeclaration::Kind::Definition:
			return AnalyzeCall(static_cast<const Function *>(function), arguments, argumentObjects);
		default:
			Utils::Logger::LogError("Unsupported function kind");
	}
}

std::vector<const DataObject *> DataObjectAnalysis::AnalyzeCall(const Function *function, const std::vector<const Operand *>& arguments, const std::vector<const DataObject *>& argumentObjects)
{
	// Collect the input data objects for the function

	Properties inputObjects;
	for (auto i = 0u; i < argumentObjects.size(); ++i)
	{
		const auto symbol = function->GetParameter(i)->GetSymbol();
		const auto object = argumentObjects.at(i);
		inputObjects[symbol] = object;
	}

	// Interprocedural analysis

	auto dataAnalysis = new DataObjectAnalysis(m_program);

	Utils::Chrono::Pause(m_functionTime);
	dataAnalysis->Analyze(function, inputObjects);
	Utils::Chrono::Continue(m_functionTime);

	m_interproceduralMap.insert({function, dataAnalysis});

	return dataAnalysis->GetReturnObjects();
}

std::vector<const DataObject *> DataObjectAnalysis::AnalyzeCall(const BuiltinFunction *function, const std::vector<const Operand *>& arguments, const std::vector<const DataObject *>& argumentObjects)
{
	switch (function->GetPrimitive())
	{
		case BuiltinFunction::Primitive::GPUOrderLib:
		{
			const auto isShared = TypeUtils::IsType<FunctionType>(arguments.at(2)->GetType());

			const auto initType = arguments.at(0)->GetType();
			const auto sortType = arguments.at(1)->GetType();
			const auto dataObject = argumentObjects.at(2 + isShared);

			const auto initFunction = TypeUtils::GetType<FunctionType>(initType)->GetFunctionDeclaration();
			const auto sortFunction = TypeUtils::GetType<FunctionType>(sortType)->GetFunctionDeclaration();

			if (argumentObjects.size() == (3 + isShared))
			{
				const auto initObjects = AnalyzeCall(initFunction, {}, {dataObject});
				const auto sortObjects = AnalyzeCall(sortFunction, {}, {initObjects.at(0), initObjects.at(1)});

				if (isShared)
				{
					const auto sharedType = arguments.at(2)->GetType();
					const auto sharedFunction = TypeUtils::GetType<FunctionType>(sharedType)->GetFunctionDeclaration();
					const auto sharedObject = AnalyzeCall(sharedFunction, {}, {initObjects.at(0), initObjects.at(1)});
				}

				return {initObjects.at(0)};
			}

			const auto orderObject = argumentObjects.at(3 + isShared);

			const auto initObjects = AnalyzeCall(initFunction, {}, {dataObject, orderObject});
			const auto sortObjects = AnalyzeCall(sortFunction, {}, {initObjects.at(0), initObjects.at(1), orderObject});

			if (isShared)
			{
				const auto sharedType = arguments.at(2)->GetType();
				const auto sharedFunction = TypeUtils::GetType<FunctionType>(sharedType)->GetFunctionDeclaration();
				const auto sharedObjects = AnalyzeCall(sharedFunction, {}, {initObjects.at(0), initObjects.at(1), orderObject});
			}

			return {initObjects.at(0)};
		}
		case BuiltinFunction::Primitive::GPUOrderInit:
		{
			return {new DataObject(), new DataObject()};
		}
		case BuiltinFunction::Primitive::GPUOrder:
		case BuiltinFunction::Primitive::GPUOrderShared:
		{
			return {};
		}
		case BuiltinFunction::Primitive::GPUGroupLib:
		{
			const auto isShared = (arguments.size() == 5);

			const auto initType = arguments.at(0)->GetType();
			const auto sortType = arguments.at(1)->GetType();
			const auto groupType = arguments.at(2 + isShared)->GetType();

			const auto initFunction = TypeUtils::GetType<FunctionType>(initType)->GetFunctionDeclaration();
			const auto sortFunction = TypeUtils::GetType<FunctionType>(sortType)->GetFunctionDeclaration();
			const auto groupFunction = TypeUtils::GetType<FunctionType>(groupType)->GetFunctionDeclaration();

			const auto initObjects = AnalyzeCall(initFunction, {}, {argumentObjects.at(3 + isShared)});
			const auto sortObjects = AnalyzeCall(sortFunction, {}, {initObjects.at(0), initObjects.at(1)});

			if (isShared)
			{
				const auto sharedType = arguments.at(2)->GetType();
				const auto sharedFunction = TypeUtils::GetType<FunctionType>(sharedType)->GetFunctionDeclaration();
				const auto sharedObjects = AnalyzeCall(sharedFunction, {}, {initObjects.at(0), initObjects.at(1)});
			}

			const auto groupObjects = AnalyzeCall(groupFunction, {}, {initObjects.at(0), initObjects.at(1)});

			return {new DataObject()};
		}
		case BuiltinFunction::Primitive::GPUGroup:
		{
			return {new DataObject(), new DataObject()};
		}
		case BuiltinFunction::Primitive::GPUUniqueLib:
		{
			const auto isShared = (arguments.size() == 5);

			const auto initType = arguments.at(0)->GetType();
			const auto sortType = arguments.at(1)->GetType();
			const auto uniqueType = arguments.at(2 + isShared)->GetType();

			const auto initFunction = TypeUtils::GetType<FunctionType>(initType)->GetFunctionDeclaration();
			const auto sortFunction = TypeUtils::GetType<FunctionType>(sortType)->GetFunctionDeclaration();
			const auto uniqueFunction = TypeUtils::GetType<FunctionType>(uniqueType)->GetFunctionDeclaration();

			const auto initObjects = AnalyzeCall(initFunction, {}, {argumentObjects.at(3 + isShared)});
			const auto sortObjects = AnalyzeCall(sortFunction, {}, {initObjects.at(0), initObjects.at(1)});

			if (isShared)
			{
				const auto sharedType = arguments.at(2)->GetType();
				const auto sharedFunction = TypeUtils::GetType<FunctionType>(sharedType)->GetFunctionDeclaration();
				const auto sharedObjects = AnalyzeCall(sharedFunction, {}, {initObjects.at(0), initObjects.at(1)});
			}

			const auto uniqueObjects = AnalyzeCall(uniqueFunction, {}, {initObjects.at(0), initObjects.at(1)});

			return {uniqueObjects.at(0)};
		}
		case BuiltinFunction::Primitive::GPUUnique:
		{
			return {new DataObject()};
		}
		case BuiltinFunction::Primitive::GPULoopJoinLib:
		{
			const auto countType = arguments.at(0)->GetType();
			const auto joinType = arguments.at(1)->GetType();

			const auto countFunction = TypeUtils::GetType<FunctionType>(countType)->GetFunctionDeclaration();
			const auto joinFunction = TypeUtils::GetType<FunctionType>(joinType)->GetFunctionDeclaration();

			// Functions are provided internally

			const auto countObjects = AnalyzeCall(countFunction, {}, {argumentObjects.at(2), argumentObjects.at(3)});
			const auto joinObjects = AnalyzeCall(joinFunction, {}, {argumentObjects.at(2), argumentObjects.at(3), countObjects.at(0), countObjects.at(1)});

			return {joinObjects.at(0)};
		}
		case BuiltinFunction::Primitive::GPUHashJoinLib:
		{
			const auto hashType = arguments.at(0)->GetType();
			const auto countType = arguments.at(1)->GetType();
			const auto joinType = arguments.at(2)->GetType();

			const auto hashFunction = TypeUtils::GetType<FunctionType>(hashType)->GetFunctionDeclaration();
			const auto countFunction = TypeUtils::GetType<FunctionType>(countType)->GetFunctionDeclaration();
			const auto joinFunction = TypeUtils::GetType<FunctionType>(joinType)->GetFunctionDeclaration();

			// Functions are provided internally

			const auto hashObjects = AnalyzeCall(hashFunction, {}, {argumentObjects.at(3)});
			const auto countObjects = AnalyzeCall(countFunction, {}, {hashObjects.at(0), argumentObjects.at(4)});
			const auto joinObjects = AnalyzeCall(joinFunction, {},
				{hashObjects.at(0), hashObjects.at(1), argumentObjects.at(4), countObjects.at(0), countObjects.at(1)}
			);

			return {joinObjects.at(0)};
		}
		case BuiltinFunction::Primitive::GPUHashCreate:
		{
			return {new DataObject(), new DataObject()};
		}
		case BuiltinFunction::Primitive::GPULoopJoinCount:
		case BuiltinFunction::Primitive::GPUHashJoinCount:
		{
			return {new DataObject(), new DataObject()};
		}
		case BuiltinFunction::Primitive::GPULoopJoin:
		case BuiltinFunction::Primitive::GPUHashJoin:
		{
			return {new DataObject()};
		}
	}
	return {new DataObject()};
}

void DataObjectAnalysis::Visit(const Identifier *identifier)
{
	// Propagate data objects through assignments

	m_expressionObjects[identifier] = {m_currentInSet.at(identifier->GetSymbol())};
}

void DataObjectAnalysis::Visit(const Literal *literal)
{
	m_expressionObjects[literal] = {new DataObject()};
}

const DataObject *DataObjectAnalysis::GetDataObject(const Operand *operand) const
{
	// Convenience for getting a single data object

	auto& dataObjects = m_expressionObjects.at(operand);
	if (dataObjects.size() > 1)
	{
		Utils::Logger::LogError("Operand has more than one data object.");
	}
	return dataObjects.at(0);
}

const std::vector<const DataObject *>& DataObjectAnalysis::GetDataObjects(const Expression *expression) const
{
	return m_expressionObjects.at(expression);
}

DataObjectAnalysis::Properties DataObjectAnalysis::InitialFlow() const
{
	// Initial flow is empty set, no data objects allocated!

	Properties initialFlow;
	return initialFlow;
}

DataObjectAnalysis::Properties DataObjectAnalysis::Merge(const Properties& s1, const Properties& s2) const
{
	// Merge the maps using set union

	Properties outSet(s1);
	for (const auto& [symbol, dataObject] : s2)
	{
		auto it = outSet.find(symbol);
		if (it != outSet.end())
		{
			if (*dataObject == *it->second)
			{
				outSet.insert({symbol, dataObject});
			}
			else
			{
				outSet[symbol] = new DataObject();
			}
		}
		else
		{
			outSet.insert({symbol, dataObject});
		}
	}
	return outSet;
}

}
}
