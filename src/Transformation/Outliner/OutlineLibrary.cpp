#include "Transformation/Outliner/OutlineLibrary.h"

#include "HorseIR/Utils/TypeUtils.h"

#include "Utils/Options.h"

namespace Transformation {

HorseIR::Statement *OutlineLibrary::Outline(const HorseIR::Statement *statement)
{
	statement->Accept(*this);
	return m_libraryStatement;
}

void OutlineLibrary::Visit(const HorseIR::Statement *statement)
{
	Utils::Logger::LogError("GPU library outliner does not support statement kind");
}

void OutlineLibrary::Visit(const HorseIR::AssignStatement *assignS)
{
	// Create the library call, copy all targets, and build the expression statement

	std::vector<HorseIR::LValue *> targets;
	for (const auto& target : assignS->GetTargets())
	{
		auto symbol = target->GetSymbol();
		m_symbols.top().insert(symbol);
		targets.push_back(new HorseIR::Identifier(symbol->name));
	}

	assignS->GetExpression()->Accept(*this);
	m_libraryStatement = new HorseIR::AssignStatement(targets, m_libraryCall);
}

void OutlineLibrary::Visit(const HorseIR::ExpressionStatement *expressionS)
{
	// Create the library call, and build the expression statement

	expressionS->GetExpression()->Accept(*this);
	m_libraryStatement = new HorseIR::ExpressionStatement(m_libraryCall);
}

void OutlineLibrary::Visit(const HorseIR::Expression *expression)
{
	Utils::Logger::LogError("GPU library outliner does not support expression kind");
}

void OutlineLibrary::Visit(const HorseIR::CallExpression *call)
{
	auto function = call->GetFunctionLiteral()->GetFunction();
	switch (function->GetKind())
	{
		case HorseIR::FunctionDeclaration::Kind::Builtin:
		{
			m_libraryCall = Outline(static_cast<const HorseIR::BuiltinFunction*>(function), call->GetArguments());
			break;
		}
		default:
		{
			Utils::Logger::LogError("Unsupported function kind");
		}
	}
}

HorseIR::CallExpression *OutlineLibrary::Outline(const HorseIR::BuiltinFunction *function, const std::vector<HorseIR::Operand *>& arguments, bool nested)
{
	switch (function->GetPrimitive())
	{
		case HorseIR::BuiltinFunction::Primitive::Each:
		{
			// Support a limited number of each functions (at the moment, only @unique)

			const auto type = arguments.at(0)->GetType();
			const auto function = HorseIR::TypeUtils::GetType<HorseIR::FunctionType>(type)->GetFunctionDeclaration();
			if (function->GetKind() != HorseIR::FunctionDeclaration::Kind::Builtin)
			{
				break;
			}

			auto builtin = static_cast<const HorseIR::BuiltinFunction *>(function);
			if (builtin->GetPrimitive() != HorseIR::BuiltinFunction::Primitive::Unique)
			{
				break;
			}
			return Outline(builtin, {arguments.at(1)}, true);
		}
		case HorseIR::BuiltinFunction::Primitive::Unique:
		{
			auto dataType = arguments.at(0)->GetType();
			auto orderLiteral = new HorseIR::BooleanLiteral({true});

			// Build the list of arguments for the library function call

			auto initFunction = GenerateInitFunction(dataType, orderLiteral, nested);
			auto sortFunction = GenerateSortFunction(dataType, orderLiteral, false, nested);

			m_functions.push_back(initFunction);
			m_functions.push_back(sortFunction);

			std::vector<HorseIR::Operand *> operands;
			operands.push_back(new HorseIR::FunctionLiteral(new HorseIR::Identifier(initFunction->GetName())));
			operands.push_back(new HorseIR::FunctionLiteral(new HorseIR::Identifier(sortFunction->GetName())));

			// Nesting disables shared memory

			if (Utils::Options::Get<bool>(Utils::Options::Opt_Algo_smem_sort) && !nested)
			{
				auto sortFunctionShared = GenerateSortFunction(dataType, orderLiteral, true);
				operands.push_back(new HorseIR::FunctionLiteral(new HorseIR::Identifier(sortFunctionShared->GetName())));
				m_functions.push_back(sortFunctionShared);
			}

			auto uniqueFunction = GenerateUniqueFunction(dataType, nested);
			m_functions.push_back(uniqueFunction);
			operands.push_back(new HorseIR::FunctionLiteral(new HorseIR::Identifier(uniqueFunction->GetName())));
			operands.push_back(arguments.at(0)->Clone());

			// Build the library call

			return new HorseIR::CallExpression(new HorseIR::FunctionLiteral(new HorseIR::Identifier("GPU", "unique_lib")), operands);
		}
		case HorseIR::BuiltinFunction::Primitive::Group:
		{
			auto dataType = arguments.at(0)->GetType();
			auto orderLiteral = new HorseIR::BooleanLiteral({true});

			// Build the list of arguments for the library function call

			auto initFunction = GenerateInitFunction(dataType, orderLiteral);
			auto sortFunction = GenerateSortFunction(dataType, orderLiteral, false);

			m_functions.push_back(initFunction);
			m_functions.push_back(sortFunction);

			std::vector<HorseIR::Operand *> operands;
			operands.push_back(new HorseIR::FunctionLiteral(new HorseIR::Identifier(initFunction->GetName())));
			operands.push_back(new HorseIR::FunctionLiteral(new HorseIR::Identifier(sortFunction->GetName())));

			if (Utils::Options::Get<bool>(Utils::Options::Opt_Algo_smem_sort))
			{
				auto sortFunctionShared = GenerateSortFunction(dataType, orderLiteral, true);
				operands.push_back(new HorseIR::FunctionLiteral(new HorseIR::Identifier(sortFunctionShared->GetName())));
				m_functions.push_back(sortFunctionShared);
			}

			auto groupFunction = GenerateGroupFunction(dataType);
			m_functions.push_back(groupFunction);
			operands.push_back(new HorseIR::FunctionLiteral(new HorseIR::Identifier(groupFunction->GetName())));
			operands.push_back(arguments.at(0)->Clone());

			// Build the library call

			return new HorseIR::CallExpression(new HorseIR::FunctionLiteral(new HorseIR::Identifier("GPU", "group_lib")), operands);
		}
		case HorseIR::BuiltinFunction::Primitive::Order:
		{
			auto dataType = arguments.at(0)->GetType();
			auto orderLiteral = dynamic_cast<const HorseIR::BooleanLiteral *>(arguments.at(1));

			// Build the list of arguments for the library function call

			auto initFunction = GenerateInitFunction(dataType, orderLiteral);
			auto sortFunction = GenerateSortFunction(dataType, orderLiteral, false);

			m_functions.push_back(initFunction);
			m_functions.push_back(sortFunction);

			std::vector<HorseIR::Operand *> operands;
			operands.push_back(new HorseIR::FunctionLiteral(new HorseIR::Identifier(initFunction->GetName())));
			operands.push_back(new HorseIR::FunctionLiteral(new HorseIR::Identifier(sortFunction->GetName())));

			if (Utils::Options::Get<bool>(Utils::Options::Opt_Algo_smem_sort))
			{
				auto sortFunctionShared = GenerateSortFunction(dataType, orderLiteral, true);
				operands.push_back(new HorseIR::FunctionLiteral(new HorseIR::Identifier(sortFunctionShared->GetName())));
				m_functions.push_back(sortFunctionShared);
			}

			operands.push_back(arguments.at(0)->Clone());
			if (orderLiteral == nullptr)
			{
				operands.push_back(arguments.at(1)->Clone());
			}

			// Build the library call

			return new HorseIR::CallExpression(new HorseIR::FunctionLiteral(new HorseIR::Identifier("GPU", "order_lib")), operands);
		}
		case HorseIR::BuiltinFunction::Primitive::JoinIndex:
		{
			std::vector<const HorseIR::Operand *> functions(std::begin(arguments), std::end(arguments) - 2);

			auto leftArgument = arguments.at(arguments.size() - 2);
			auto rightArgument = arguments.at(arguments.size() - 1);

			auto leftType = leftArgument->GetType();
			auto rightType = rightArgument->GetType();

			std::vector<HorseIR::Operand *> operands;

			auto isHashing = false;
			switch (Utils::Options::GetJoinKind())
			{
				case Utils::Options::JoinKind::LoopJoin:
				{
					isHashing = false;
					break;
				}
				case Utils::Options::JoinKind::HashJoin:
				{
					isHashing = true;
					for (const auto function : functions)
					{
						const auto functionType = HorseIR::TypeUtils::GetType<HorseIR::FunctionType>(function->GetType());
						const auto declaration = functionType->GetFunctionDeclaration();

						if (declaration->GetKind() != HorseIR::FunctionDeclaration::Kind::Builtin)
						{
							isHashing = false;
							break;
						}

						auto primitive = static_cast<const HorseIR::BuiltinFunction *>(declaration);
						if (primitive->GetPrimitive() != HorseIR::BuiltinFunction::Primitive::Equal)
						{
							isHashing = false;
							break;
						}
					}

					if (isHashing)
					{
						auto hashFunction = GenerateHashFunction(leftType);
						m_functions.push_back(hashFunction);
						operands.push_back(new HorseIR::FunctionLiteral(new HorseIR::Identifier(hashFunction->GetName())));
					}
					break;
				}
			}

			auto countFunction = GenerateJoinCountFunction(functions, leftType, rightType, isHashing);
			auto joinFunction = GenerateJoinFunction(functions, leftType, rightType, isHashing);

			m_functions.push_back(countFunction);
			m_functions.push_back(joinFunction);

			// Build the list of arguments for the library function call

			operands.push_back(new HorseIR::FunctionLiteral(new HorseIR::Identifier(countFunction->GetName())));
			operands.push_back(new HorseIR::FunctionLiteral(new HorseIR::Identifier(joinFunction->GetName())));

			operands.push_back(leftArgument->Clone());
			operands.push_back(rightArgument->Clone());

			// Build the library call

			auto name = (isHashing) ? "hash_join_lib" : "loop_join_lib";
			return new HorseIR::CallExpression(new HorseIR::FunctionLiteral(new HorseIR::Identifier("GPU", name)), operands);
		}
	}
	Utils::Logger::LogError("GPU library outliner does not support builtin function '" + function->GetName() + "'");
}

HorseIR::Function *OutlineLibrary::GenerateInitFunction(const HorseIR::Type *dataType, const HorseIR::BooleanLiteral *orders, bool nested)
{
	std::vector<HorseIR::Parameter *> parameters;

	std::vector<HorseIR::Operand *> operands;
	std::vector<HorseIR::LValue *> lvalues;

	std::vector<HorseIR::Operand *> returnOperands;
	std::vector<HorseIR::Type *> returnTypes;

	parameters.push_back(new HorseIR::Parameter("data", dataType->Clone()));
	operands.push_back(new HorseIR::Identifier("data"));

	if (orders == nullptr)
	{
		parameters.push_back(new HorseIR::Parameter("orders", new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Boolean)));
		operands.push_back(new HorseIR::Identifier("orders"));
	}
	else
	{
		operands.push_back(orders->Clone());
	}

	HorseIR::Type *indexType = new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64);
	if (nested)
	{
		indexType = new HorseIR::ListType(indexType);
	}

	lvalues.push_back(new HorseIR::VariableDeclaration("index", indexType));
	returnTypes.push_back(indexType->Clone());
	returnOperands.push_back(new HorseIR::Identifier("index"));

	lvalues.push_back(new HorseIR::VariableDeclaration("data_out", dataType->Clone()));
	returnOperands.push_back(new HorseIR::Identifier("data_out"));
	returnTypes.push_back(dataType->Clone());

	auto funcLiteral = new HorseIR::FunctionLiteral(new HorseIR::Identifier("GPU", "order_init"));
	if (nested)
	{
		operands.insert(std::begin(operands), funcLiteral);
		funcLiteral = new HorseIR::FunctionLiteral(new HorseIR::Identifier("each_left"));
	}

	auto initCall = new HorseIR::CallExpression(funcLiteral, operands);
	auto initStatement = new HorseIR::AssignStatement(lvalues, initCall);
	auto returnStatement = new HorseIR::ReturnStatement(returnOperands);

	return new HorseIR::Function("order_init_" + std::to_string(m_index++), parameters, returnTypes, {initStatement, returnStatement}, true);
}

HorseIR::Function *OutlineLibrary::GenerateSortFunction(const HorseIR::Type *dataType, const HorseIR::BooleanLiteral *orders, bool shared, bool nested)
{
	std::vector<HorseIR::Parameter *> parameters;
	std::vector<HorseIR::Operand *> operands;

	HorseIR::Type *indexType = new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64);
	if (nested)
	{
		indexType = new HorseIR::ListType(indexType);
	}

	parameters.push_back(new HorseIR::Parameter("index", indexType));
	operands.push_back(new HorseIR::Identifier("index"));

	parameters.push_back(new HorseIR::Parameter("data", dataType->Clone()));
	operands.push_back(new HorseIR::Identifier("data"));

	if (orders == nullptr)
	{
		parameters.push_back(new HorseIR::Parameter("orders", new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Boolean)));
		operands.push_back(new HorseIR::Identifier("orders"));
	}
	else
	{
		operands.push_back(orders->Clone());
	}

	auto name = std::string((shared) ? "order_shared" : "order");
	auto funcLiteral = new HorseIR::FunctionLiteral(new HorseIR::Identifier("GPU", name));
	if (nested)
	{
		operands.insert(std::begin(operands), funcLiteral);
		funcLiteral = new HorseIR::FunctionLiteral(new HorseIR::Identifier("each_item"));
	}

	auto sortCall = new HorseIR::CallExpression(funcLiteral, operands);
	auto sortStatement = new HorseIR::ExpressionStatement(sortCall);
	
	return new HorseIR::Function(name + "_" + std::to_string(m_index++), parameters, {}, {sortStatement}, true);
}

HorseIR::Function *OutlineLibrary::GenerateGroupFunction(const HorseIR::Type *dataType)
{
	std::vector<HorseIR::Parameter *> parameters;

	std::vector<HorseIR::Operand *> operands;
	std::vector<HorseIR::LValue *> lvalues;

	std::vector<HorseIR::Operand *> returnOperands;
	std::vector<HorseIR::Type *> returnTypes;

	parameters.push_back(new HorseIR::Parameter("index", new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64)));
	operands.push_back(new HorseIR::Identifier("index"));

	parameters.push_back(new HorseIR::Parameter("data", dataType->Clone()));
	operands.push_back(new HorseIR::Identifier("data"));

	lvalues.push_back(new HorseIR::VariableDeclaration("keys", new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64)));
	lvalues.push_back(new HorseIR::VariableDeclaration("values", new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64)));

	returnOperands.push_back(new HorseIR::Identifier("keys"));
	returnOperands.push_back(new HorseIR::Identifier("values"));

	returnTypes.push_back(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64));
	returnTypes.push_back(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64));

	auto groupCall = new HorseIR::CallExpression(new HorseIR::FunctionLiteral(new HorseIR::Identifier("GPU", "group")), operands);
	auto groupStatement = new HorseIR::AssignStatement(lvalues, groupCall);
	auto returnStatement = new HorseIR::ReturnStatement(returnOperands);

	return new HorseIR::Function("group_" + std::to_string(m_index++), parameters, returnTypes, {groupStatement, returnStatement}, true);
}

HorseIR::Function *OutlineLibrary::GenerateUniqueFunction(const HorseIR::Type *dataType, bool nested)
{
	std::vector<HorseIR::Parameter *> parameters;

	std::vector<HorseIR::Operand *> operands;
	std::vector<HorseIR::LValue *> lvalues;

	std::vector<HorseIR::Operand *> returnOperands;
	std::vector<HorseIR::Type *> returnTypes;

	HorseIR::Type *indexType = new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64);
	if (nested)
	{
		indexType = new HorseIR::ListType(indexType);
	}

	parameters.push_back(new HorseIR::Parameter("index", indexType));
	operands.push_back(new HorseIR::Identifier("index"));

	parameters.push_back(new HorseIR::Parameter("data", dataType->Clone()));
	operands.push_back(new HorseIR::Identifier("data"));

	lvalues.push_back(new HorseIR::VariableDeclaration("keys", indexType->Clone()));
	returnTypes.push_back(indexType->Clone());
	returnOperands.push_back(new HorseIR::Identifier("keys"));

	auto funcLiteral = new HorseIR::FunctionLiteral(new HorseIR::Identifier("GPU", "unique"));
	if (nested)
	{
		operands.insert(std::begin(operands), funcLiteral);
		funcLiteral = new HorseIR::FunctionLiteral(new HorseIR::Identifier("each_item"));
	}

	auto uniqueCall = new HorseIR::CallExpression(funcLiteral, operands);
	auto uniqueStatement = new HorseIR::AssignStatement(lvalues, uniqueCall);
	auto returnStatement = new HorseIR::ReturnStatement(returnOperands);

	return new HorseIR::Function("unique_" + std::to_string(m_index++), parameters, returnTypes, {uniqueStatement, returnStatement}, true);
}

HorseIR::Function *OutlineLibrary::GenerateHashFunction(const HorseIR::Type *dataType)
{
	std::vector<HorseIR::Parameter *> parameters;

	std::vector<HorseIR::Operand *> operands;
	std::vector<HorseIR::LValue *> lvalues;

	std::vector<HorseIR::Operand *> returnOperands;
	std::vector<HorseIR::Type *> returnTypes;

	parameters.push_back(new HorseIR::Parameter("data", dataType->Clone()));
	operands.push_back(new HorseIR::Identifier("data"));

	lvalues.push_back(new HorseIR::VariableDeclaration("keys", dataType->Clone()));
	returnOperands.push_back(new HorseIR::Identifier("keys"));
	returnTypes.push_back(dataType->Clone());

	lvalues.push_back(new HorseIR::VariableDeclaration("values", new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64)));
	returnOperands.push_back(new HorseIR::Identifier("values"));
	returnTypes.push_back(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64));

	auto hashCall = new HorseIR::CallExpression(new HorseIR::FunctionLiteral(new HorseIR::Identifier("GPU", "hash_create")), operands);
	auto hashStatement = new HorseIR::AssignStatement(lvalues, hashCall);
	auto returnStatement = new HorseIR::ReturnStatement(returnOperands);

	return new HorseIR::Function("hash_create_" + std::to_string(m_index++), parameters, returnTypes, {hashStatement, returnStatement}, true);
}


HorseIR::Function *OutlineLibrary::GenerateJoinCountFunction(std::vector<const HorseIR::Operand *>& functions, const HorseIR::Type *leftType, const HorseIR::Type *rightType, bool isHashing)
{
	std::vector<HorseIR::Parameter *> parameters;

	std::vector<HorseIR::Operand *> operands;
	std::vector<HorseIR::LValue *> lvalues;

	std::vector<HorseIR::Operand *> returnOperands;
	std::vector<HorseIR::Type *> returnTypes;

	if (!isHashing)
	{
		for (const auto argument : functions)
		{
			operands.push_back(argument->Clone());
		}
	}

	parameters.push_back(new HorseIR::Parameter("data_left", leftType->Clone()));
	operands.push_back(new HorseIR::Identifier("data_left"));

	if (isHashing)
	{
		parameters.push_back(new HorseIR::Parameter("hash_keys", leftType->Clone()));
		operands.push_back(new HorseIR::Identifier("hash_keys"));
	}
	else
	{
		parameters.push_back(new HorseIR::Parameter("data_right", rightType->Clone()));
		operands.push_back(new HorseIR::Identifier("data_right"));
	}

	lvalues.push_back(new HorseIR::VariableDeclaration("offsets", new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64)));
	lvalues.push_back(new HorseIR::VariableDeclaration("count", new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64)));

	returnOperands.push_back(new HorseIR::Identifier("offsets"));
	returnOperands.push_back(new HorseIR::Identifier("count"));
	returnTypes.push_back(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64));
	returnTypes.push_back(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64));

	auto joinName = (isHashing) ? "hash_join_count" : "loop_join_count";
	auto joinCall = new HorseIR::CallExpression(new HorseIR::FunctionLiteral(new HorseIR::Identifier("GPU", joinName)), operands);
	auto joinStatement = new HorseIR::AssignStatement(lvalues, joinCall);
	auto returnStatement = new HorseIR::ReturnStatement(returnOperands);

	return new HorseIR::Function("join_count_" + std::to_string(m_index++), parameters, returnTypes, {joinStatement, returnStatement}, true);
}

HorseIR::Function *OutlineLibrary::GenerateJoinFunction(std::vector<const HorseIR::Operand *>& functions, const HorseIR::Type *leftType, const HorseIR::Type *rightType, bool isHashing)
{
	std::vector<HorseIR::Parameter *> parameters;

	std::vector<HorseIR::Operand *> operands;
	std::vector<HorseIR::LValue *> lvalues;

	std::vector<HorseIR::Operand *> returnOperands;
	std::vector<HorseIR::Type *> returnTypes;

	if (!isHashing)
	{
		for (const auto argument : functions)
		{
			operands.push_back(argument->Clone());
		}
	}

	parameters.push_back(new HorseIR::Parameter("data_left", leftType->Clone()));
	operands.push_back(new HorseIR::Identifier("data_left"));

	if (isHashing)
	{
		parameters.push_back(new HorseIR::Parameter("hash_keys", leftType->Clone()));
		operands.push_back(new HorseIR::Identifier("hash_keys"));

		parameters.push_back(new HorseIR::Parameter("hash_values", new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64)));
		operands.push_back(new HorseIR::Identifier("hash_values"));
	}
	else
	{
		parameters.push_back(new HorseIR::Parameter("data_right", rightType->Clone()));
		operands.push_back(new HorseIR::Identifier("data_right"));
	}

	parameters.push_back(new HorseIR::Parameter("offsets", new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64)));
	operands.push_back(new HorseIR::Identifier("offsets"));

	parameters.push_back(new HorseIR::Parameter("count", new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64)));
	operands.push_back(new HorseIR::Identifier("count"));

	lvalues.push_back(new HorseIR::VariableDeclaration("indexes", new HorseIR::ListType(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64))));

	returnOperands.push_back(new HorseIR::Identifier("indexes"));
	returnTypes.push_back(new HorseIR::ListType(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64)));

	auto joinName = (isHashing) ? "hash_join" : "loop_join";
	auto joinCall = new HorseIR::CallExpression(new HorseIR::FunctionLiteral(new HorseIR::Identifier("GPU", joinName)), operands);
	auto joinStatement = new HorseIR::AssignStatement(lvalues, joinCall);
	auto returnStatement = new HorseIR::ReturnStatement(returnOperands);

	return new HorseIR::Function("join_" + std::to_string(m_index++), parameters, returnTypes, {joinStatement, returnStatement}, true);
}

}
