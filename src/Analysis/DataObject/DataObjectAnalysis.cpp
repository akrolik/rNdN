#include "Analysis/DataObject/DataObjectAnalysis.h"

namespace Analysis {

void DataObjectAnalysis::Visit(const HorseIR::Parameter *parameter)
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

void DataObjectAnalysis::Visit(const HorseIR::AssignStatement *assignS)
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

void DataObjectAnalysis::Visit(const HorseIR::BlockStatement *blockS)
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

void DataObjectAnalysis::Visit(const HorseIR::ReturnStatement *returnS)
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

void DataObjectAnalysis::Visit(const HorseIR::CastExpression *cast)
{
	// Create new data objects for other (non-identifier) expressions on RHS of assignment

	m_expressionObjects[cast] = {new DataObject()};
}

void DataObjectAnalysis::Visit(const HorseIR::CallExpression *call)
{
	// Accumulate data objects for arguments

	std::vector<const DataObject *> argumentObjects;
	for (const auto argument : call->GetArguments())
	{
		argument->Accept(*this);
		argumentObjects.push_back(GetDataObject(argument));
	}
	m_expressionObjects[call] = AnalyzeCall(call->GetFunctionLiteral()->GetFunction(), argumentObjects);
}

std::vector<const DataObject *> DataObjectAnalysis::AnalyzeCall(const HorseIR::FunctionDeclaration *function, const std::vector<const DataObject *>& argumentObjects)
{
	switch (function->GetKind())
	{
		case HorseIR::FunctionDeclaration::Kind::Builtin:
			return AnalyzeCall(static_cast<const HorseIR::BuiltinFunction *>(function), argumentObjects);
		case HorseIR::FunctionDeclaration::Kind::Definition:
			return AnalyzeCall(static_cast<const HorseIR::Function *>(function), argumentObjects);
		default:
			Utils::Logger::LogError("Unsupported function kind");
	}
}

std::vector<const DataObject *> DataObjectAnalysis::AnalyzeCall(const HorseIR::Function *function, const std::vector<const DataObject *>& argumentObjects)
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
	dataAnalysis->Analyze(function, inputObjects);

	m_interproceduralMap.insert({function, dataAnalysis});

	return dataAnalysis->GetReturnObjects();
}

std::vector<const DataObject *> DataObjectAnalysis::AnalyzeCall(const HorseIR::BuiltinFunction *function, const std::vector<const DataObject *>& argumentObjects)
{
	return {new DataObject()};
}

void DataObjectAnalysis::Visit(const HorseIR::Identifier *identifier)
{
	// Propagate data objects through assignments

	m_expressionObjects[identifier] = {m_currentInSet.at(identifier->GetSymbol())};
}

void DataObjectAnalysis::Visit(const HorseIR::Literal *literal)
{
	m_expressionObjects[literal] = {new DataObject()};
}

const DataObject *DataObjectAnalysis::GetDataObject(const HorseIR::Operand *operand) const
{
	// Convenience for getting a single data object

	auto& dataObjects = m_expressionObjects.at(operand);
	if (dataObjects.size() > 1)
	{
		Utils::Logger::LogError("Operand has more than one data object.");
	}
	return dataObjects.at(0);
}

const std::vector<const DataObject *>& DataObjectAnalysis::GetDataObjects(const HorseIR::Expression *expression) const
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
