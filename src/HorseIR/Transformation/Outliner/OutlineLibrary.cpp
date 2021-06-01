#include "HorseIR/Transformation/Outliner/OutlineLibrary.h"

#include "HorseIR/Utils/TypeUtils.h"

#include "Utils/Options.h"

namespace HorseIR {
namespace Transformation {

Statement *OutlineLibrary::Outline(const Statement *statement)
{
	statement->Accept(*this);
	return m_libraryStatement;
}

void OutlineLibrary::Visit(const Statement *statement)
{
	Utils::Logger::LogError("GPU library outliner does not support statement kind");
}

void OutlineLibrary::Visit(const AssignStatement *assignS)
{
	// Create the library call, copy all targets, and build the expression statement

	std::vector<LValue *> targets;
	for (const auto& target : assignS->GetTargets())
	{
		auto symbol = target->GetSymbol();
		m_symbols.top().insert(symbol);
		targets.push_back(new Identifier(symbol->name));
	}

	assignS->GetExpression()->Accept(*this);
	m_libraryStatement = new AssignStatement(targets, m_libraryCall);
}

void OutlineLibrary::Visit(const ExpressionStatement *expressionS)
{
	// Create the library call, and build the expression statement

	expressionS->GetExpression()->Accept(*this);
	m_libraryStatement = new ExpressionStatement(m_libraryCall);
}

void OutlineLibrary::Visit(const Expression *expression)
{
	Utils::Logger::LogError("GPU library outliner does not support expression kind");
}

void OutlineLibrary::Visit(const CallExpression *call)
{
	auto function = call->GetFunctionLiteral()->GetFunction();
	switch (function->GetKind())
	{
		case FunctionDeclaration::Kind::Builtin:
		{
			m_libraryCall = Outline(static_cast<const BuiltinFunction*>(function), call->GetArguments());
			break;
		}
		default:
		{
			Utils::Logger::LogError("Unsupported function kind");
		}
	}
}

CallExpression *OutlineLibrary::Outline(const BuiltinFunction *function, const std::vector<const Operand *>& arguments, bool nested)
{
	switch (function->GetPrimitive())
	{
		case BuiltinFunction::Primitive::Each:
		{
			// Support a limited number of each functions (at the moment, only @unique)

			const auto type = arguments.at(0)->GetType();
			const auto function = TypeUtils::GetType<FunctionType>(type)->GetFunctionDeclaration();
			if (function->GetKind() != FunctionDeclaration::Kind::Builtin)
			{
				break;
			}

			auto builtin = static_cast<const BuiltinFunction *>(function);
			if (builtin->GetPrimitive() != BuiltinFunction::Primitive::Unique)
			{
				break;
			}
			return Outline(builtin, {arguments.at(1)}, true);
		}
		case BuiltinFunction::Primitive::Unique:
		{
			auto dataType = arguments.at(0)->GetType();
			auto orderLiteral = new BooleanLiteral({true});

			// Build the list of arguments for the library function call

			auto initFunction = GenerateInitFunction(dataType, orderLiteral, nested);
			auto sortFunction = GenerateSortFunction(dataType, orderLiteral, false, nested);

			m_functions.push_back(initFunction);
			m_functions.push_back(sortFunction);

			std::vector<Operand *> operands;
			operands.push_back(new FunctionLiteral(new Identifier(initFunction->GetName())));
			operands.push_back(new FunctionLiteral(new Identifier(sortFunction->GetName())));

			// Nesting disables shared memory

			if (Utils::Options::GetAlgorithm_SortKind() == Utils::Options::SortKind::SharedSort && !nested)
			{
				auto sortFunctionShared = GenerateSortFunction(dataType, orderLiteral, true);
				operands.push_back(new FunctionLiteral(new Identifier(sortFunctionShared->GetName())));
				m_functions.push_back(sortFunctionShared);
			}

			auto uniqueFunction = GenerateUniqueFunction(dataType, nested);
			m_functions.push_back(uniqueFunction);
			operands.push_back(new FunctionLiteral(new Identifier(uniqueFunction->GetName())));
			operands.push_back(arguments.at(0)->Clone());

			// Build the library call

			return new CallExpression(new FunctionLiteral(new Identifier("GPU", "unique_lib")), operands);
		}
		case BuiltinFunction::Primitive::Group:
		{
			auto dataType = arguments.at(0)->GetType();
			auto orderLiteral = new BooleanLiteral({true});

			// Build the list of arguments for the library function call

			auto initFunction = GenerateInitFunction(dataType, orderLiteral);
			auto sortFunction = GenerateSortFunction(dataType, orderLiteral, false);

			m_functions.push_back(initFunction);
			m_functions.push_back(sortFunction);

			std::vector<Operand *> operands;
			operands.push_back(new FunctionLiteral(new Identifier(initFunction->GetName())));
			operands.push_back(new FunctionLiteral(new Identifier(sortFunction->GetName())));

			if (Utils::Options::GetAlgorithm_SortKind() == Utils::Options::SortKind::SharedSort)
			{
				auto sortFunctionShared = GenerateSortFunction(dataType, orderLiteral, true);
				operands.push_back(new FunctionLiteral(new Identifier(sortFunctionShared->GetName())));
				m_functions.push_back(sortFunctionShared);
			}

			auto groupFunction = GenerateGroupFunction(dataType);
			m_functions.push_back(groupFunction);
			operands.push_back(new FunctionLiteral(new Identifier(groupFunction->GetName())));
			operands.push_back(arguments.at(0)->Clone());

			// Build the library call

			return new CallExpression(new FunctionLiteral(new Identifier("GPU", "group_lib")), operands);
		}
		case BuiltinFunction::Primitive::Order:
		{
			auto dataType = arguments.at(0)->GetType();
			auto orderLiteral = dynamic_cast<const BooleanLiteral *>(arguments.at(1));

			// Build the list of arguments for the library function call

			auto initFunction = GenerateInitFunction(dataType, orderLiteral);
			auto sortFunction = GenerateSortFunction(dataType, orderLiteral, false);

			m_functions.push_back(initFunction);
			m_functions.push_back(sortFunction);

			std::vector<Operand *> operands;
			operands.push_back(new FunctionLiteral(new Identifier(initFunction->GetName())));
			operands.push_back(new FunctionLiteral(new Identifier(sortFunction->GetName())));

			if (Utils::Options::GetAlgorithm_SortKind() == Utils::Options::SortKind::SharedSort)
			{
				auto sortFunctionShared = GenerateSortFunction(dataType, orderLiteral, true);
				operands.push_back(new FunctionLiteral(new Identifier(sortFunctionShared->GetName())));
				m_functions.push_back(sortFunctionShared);
			}

			operands.push_back(arguments.at(0)->Clone());
			if (orderLiteral == nullptr)
			{
				operands.push_back(arguments.at(1)->Clone());
			}

			// Build the library call

			return new CallExpression(new FunctionLiteral(new Identifier("GPU", "order_lib")), operands);
		}
		case BuiltinFunction::Primitive::JoinIndex:
		{
			std::vector<const Operand *> functions(std::begin(arguments), std::end(arguments) - 2);

			auto leftArgument = arguments.at(arguments.size() - 2);
			auto rightArgument = arguments.at(arguments.size() - 1);

			auto leftType = leftArgument->GetType();
			auto rightType = rightArgument->GetType();

			std::vector<Operand *> operands;

			auto isHashing = false;
			switch (Utils::Options::GetAlgorithm_JoinKind())
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
						const auto functionType = TypeUtils::GetType<FunctionType>(function->GetType());
						const auto declaration = functionType->GetFunctionDeclaration();

						if (declaration->GetKind() != FunctionDeclaration::Kind::Builtin)
						{
							isHashing = false;
							break;
						}

						auto primitive = static_cast<const BuiltinFunction *>(declaration);
						if (primitive->GetPrimitive() != BuiltinFunction::Primitive::Equal)
						{
							isHashing = false;
							break;
						}
					}

					if (isHashing)
					{
						auto hashFunction = GenerateHashFunction(leftType);
						m_functions.push_back(hashFunction);
						operands.push_back(new FunctionLiteral(new Identifier(hashFunction->GetName())));
					}
					break;
				}
			}

			auto countFunction = GenerateJoinCountFunction(functions, leftType, rightType, isHashing);
			auto joinFunction = GenerateJoinFunction(functions, leftType, rightType, isHashing);

			m_functions.push_back(countFunction);
			m_functions.push_back(joinFunction);

			// Build the list of arguments for the library function call

			operands.push_back(new FunctionLiteral(new Identifier(countFunction->GetName())));
			operands.push_back(new FunctionLiteral(new Identifier(joinFunction->GetName())));

			operands.push_back(leftArgument->Clone());
			operands.push_back(rightArgument->Clone());

			// Build the library call

			auto name = (isHashing) ? "hash_join_lib" : "loop_join_lib";
			return new CallExpression(new FunctionLiteral(new Identifier("GPU", name)), operands);
		}
		case BuiltinFunction::Primitive::Member:
		{
			auto leftArgument = arguments.at(0);
			auto rightArgument = arguments.at(1);

			auto leftType = leftArgument->GetType();
			auto rightType = rightArgument->GetType();

			// Build the library functions

			auto hashFunction = GenerateMemberHashFunction(rightType);
			auto memberFunction = GenerateMemberFunction(rightType, leftType);

			m_functions.push_back(hashFunction);
			m_functions.push_back(memberFunction);

			// Build the list of arguments for the library function call

			std::vector<Operand *> operands;

			operands.push_back(new FunctionLiteral(new Identifier(hashFunction->GetName())));
			operands.push_back(new FunctionLiteral(new Identifier(memberFunction->GetName())));

			operands.push_back(leftArgument->Clone());
			operands.push_back(rightArgument->Clone());

			// Build the library call

			return new CallExpression(new FunctionLiteral(new Identifier("GPU", "hash_member_lib")), operands);
		}
	}
	Utils::Logger::LogError("GPU library outliner does not support builtin function '" + function->GetName() + "'");
}

Function *OutlineLibrary::GenerateInitFunction(const Type *dataType, const BooleanLiteral *orders, bool nested)
{
	std::vector<Parameter *> parameters;

	std::vector<Operand *> operands;
	std::vector<LValue *> lvalues;

	std::vector<Operand *> returnOperands;
	std::vector<Type *> returnTypes;

	parameters.push_back(new Parameter("data", dataType->Clone()));
	operands.push_back(new Identifier("data"));

	if (orders == nullptr)
	{
		parameters.push_back(new Parameter("orders", new BasicType(BasicType::BasicKind::Boolean)));
		operands.push_back(new Identifier("orders"));
	}
	else
	{
		operands.push_back(orders->Clone());
	}

	Type *indexType = new BasicType(BasicType::BasicKind::Int64);
	if (nested)
	{
		indexType = new ListType(indexType);
	}

	lvalues.push_back(new VariableDeclaration("index", indexType));
	returnTypes.push_back(indexType->Clone());
	returnOperands.push_back(new Identifier("index"));

	lvalues.push_back(new VariableDeclaration("data_out", dataType->Clone()));
	returnOperands.push_back(new Identifier("data_out"));
	returnTypes.push_back(dataType->Clone());

	auto funcLiteral = new FunctionLiteral(new Identifier("GPU", "order_init"));
	if (nested)
	{
		operands.insert(std::begin(operands), funcLiteral);
		funcLiteral = new FunctionLiteral(new Identifier("each_left"));
	}

	auto initCall = new CallExpression(funcLiteral, operands);
	auto initStatement = new AssignStatement(lvalues, initCall);
	auto returnStatement = new ReturnStatement(returnOperands);

	return new Function("order_init_" + std::to_string(m_index++), parameters, returnTypes, {initStatement, returnStatement}, true);
}

Function *OutlineLibrary::GenerateSortFunction(const Type *dataType, const BooleanLiteral *orders, bool shared, bool nested)
{
	std::vector<Parameter *> parameters;
	std::vector<Operand *> operands;

	Type *indexType = new BasicType(BasicType::BasicKind::Int64);
	if (nested)
	{
		indexType = new ListType(indexType);
	}

	parameters.push_back(new Parameter("index", indexType));
	operands.push_back(new Identifier("index"));

	parameters.push_back(new Parameter("data", dataType->Clone()));
	operands.push_back(new Identifier("data"));

	if (orders == nullptr)
	{
		parameters.push_back(new Parameter("orders", new BasicType(BasicType::BasicKind::Boolean)));
		operands.push_back(new Identifier("orders"));
	}
	else
	{
		operands.push_back(orders->Clone());
	}

	auto name = std::string((shared) ? "order_shared" : "order");
	auto funcLiteral = new FunctionLiteral(new Identifier("GPU", name));
	if (nested)
	{
		operands.insert(std::begin(operands), funcLiteral);
		funcLiteral = new FunctionLiteral(new Identifier("each_item"));
	}

	auto sortCall = new CallExpression(funcLiteral, operands);
	auto sortStatement = new ExpressionStatement(sortCall);
	
	return new Function(name + "_" + std::to_string(m_index++), parameters, {}, {sortStatement}, true);
}

Function *OutlineLibrary::GenerateGroupFunction(const Type *dataType)
{
	std::vector<Parameter *> parameters;

	std::vector<Operand *> operands;
	std::vector<LValue *> lvalues;

	std::vector<Operand *> returnOperands;
	std::vector<Type *> returnTypes;

	parameters.push_back(new Parameter("index", new BasicType(BasicType::BasicKind::Int64)));
	operands.push_back(new Identifier("index"));

	parameters.push_back(new Parameter("data", dataType->Clone()));
	operands.push_back(new Identifier("data"));

	lvalues.push_back(new VariableDeclaration("keys", new BasicType(BasicType::BasicKind::Int64)));
	lvalues.push_back(new VariableDeclaration("values", new BasicType(BasicType::BasicKind::Int64)));

	returnOperands.push_back(new Identifier("keys"));
	returnOperands.push_back(new Identifier("values"));

	returnTypes.push_back(new BasicType(BasicType::BasicKind::Int64));
	returnTypes.push_back(new BasicType(BasicType::BasicKind::Int64));

	auto groupCall = new CallExpression(new FunctionLiteral(new Identifier("GPU", "group")), operands);
	auto groupStatement = new AssignStatement(lvalues, groupCall);
	auto returnStatement = new ReturnStatement(returnOperands);

	return new Function("group_" + std::to_string(m_index++), parameters, returnTypes, {groupStatement, returnStatement}, true);
}

Function *OutlineLibrary::GenerateUniqueFunction(const Type *dataType, bool nested)
{
	std::vector<Parameter *> parameters;

	std::vector<Operand *> operands;
	std::vector<LValue *> lvalues;

	std::vector<Operand *> returnOperands;
	std::vector<Type *> returnTypes;

	Type *indexType = new BasicType(BasicType::BasicKind::Int64);
	if (nested)
	{
		indexType = new ListType(indexType);
	}

	parameters.push_back(new Parameter("index", indexType));
	operands.push_back(new Identifier("index"));

	parameters.push_back(new Parameter("data", dataType->Clone()));
	operands.push_back(new Identifier("data"));

	lvalues.push_back(new VariableDeclaration("keys", indexType->Clone()));
	returnTypes.push_back(indexType->Clone());
	returnOperands.push_back(new Identifier("keys"));

	auto funcLiteral = new FunctionLiteral(new Identifier("GPU", "unique"));
	if (nested)
	{
		operands.insert(std::begin(operands), funcLiteral);
		funcLiteral = new FunctionLiteral(new Identifier("each_item"));
	}

	auto uniqueCall = new CallExpression(funcLiteral, operands);
	auto uniqueStatement = new AssignStatement(lvalues, uniqueCall);
	auto returnStatement = new ReturnStatement(returnOperands);

	return new Function("unique_" + std::to_string(m_index++), parameters, returnTypes, {uniqueStatement, returnStatement}, true);
}

Function *OutlineLibrary::GenerateHashFunction(const Type *dataType)
{
	std::vector<Parameter *> parameters;

	std::vector<Operand *> operands;
	std::vector<LValue *> lvalues;

	std::vector<Operand *> returnOperands;
	std::vector<Type *> returnTypes;

	parameters.push_back(new Parameter("data", dataType->Clone()));
	operands.push_back(new Identifier("data"));

	lvalues.push_back(new VariableDeclaration("keys", dataType->Clone()));
	returnOperands.push_back(new Identifier("keys"));
	returnTypes.push_back(dataType->Clone());

	lvalues.push_back(new VariableDeclaration("values", new BasicType(BasicType::BasicKind::Int64)));
	returnOperands.push_back(new Identifier("values"));
	returnTypes.push_back(new BasicType(BasicType::BasicKind::Int64));

	auto hashCall = new CallExpression(new FunctionLiteral(new Identifier("GPU", "hash_create")), operands);
	auto hashStatement = new AssignStatement(lvalues, hashCall);
	auto returnStatement = new ReturnStatement(returnOperands);

	return new Function("hash_create_" + std::to_string(m_index++), parameters, returnTypes, {hashStatement, returnStatement}, true);
}


Function *OutlineLibrary::GenerateJoinCountFunction(std::vector<const Operand *>& functions, const Type *leftType, const Type *rightType, bool isHashing)
{
	std::vector<Parameter *> parameters;

	std::vector<Operand *> operands;
	std::vector<LValue *> lvalues;

	std::vector<Operand *> returnOperands;
	std::vector<Type *> returnTypes;

	if (isHashing)
	{
		parameters.push_back(new Parameter("hash_keys", leftType->Clone()));
		operands.push_back(new Identifier("hash_keys"));
	}
	else
	{
		for (const auto argument : functions)
		{
			operands.push_back(argument->Clone());
		}

		parameters.push_back(new Parameter("data_left", leftType->Clone()));
		operands.push_back(new Identifier("data_left"));
	}

	parameters.push_back(new Parameter("data_right", rightType->Clone()));
	operands.push_back(new Identifier("data_right"));

	lvalues.push_back(new VariableDeclaration("offsets", new BasicType(BasicType::BasicKind::Int64)));
	lvalues.push_back(new VariableDeclaration("count", new BasicType(BasicType::BasicKind::Int64)));

	returnOperands.push_back(new Identifier("offsets"));
	returnOperands.push_back(new Identifier("count"));
	returnTypes.push_back(new BasicType(BasicType::BasicKind::Int64));
	returnTypes.push_back(new BasicType(BasicType::BasicKind::Int64));

	auto joinName = (isHashing) ? "hash_join_count" : "loop_join_count";
	auto joinCall = new CallExpression(new FunctionLiteral(new Identifier("GPU", joinName)), operands);
	auto joinStatement = new AssignStatement(lvalues, joinCall);
	auto returnStatement = new ReturnStatement(returnOperands);

	return new Function("join_count_" + std::to_string(m_index++), parameters, returnTypes, {joinStatement, returnStatement}, true);
}

Function *OutlineLibrary::GenerateJoinFunction(std::vector<const Operand *>& functions, const Type *leftType, const Type *rightType, bool isHashing)
{
	std::vector<Parameter *> parameters;

	std::vector<Operand *> operands;
	std::vector<LValue *> lvalues;

	std::vector<Operand *> returnOperands;
	std::vector<Type *> returnTypes;

	if (isHashing)
	{
		parameters.push_back(new Parameter("hash_keys", leftType->Clone()));
		operands.push_back(new Identifier("hash_keys"));

		parameters.push_back(new Parameter("hash_values", new BasicType(BasicType::BasicKind::Int64)));
		operands.push_back(new Identifier("hash_values"));
	}
	else
	{
		for (const auto argument : functions)
		{
			operands.push_back(argument->Clone());
		}

		parameters.push_back(new Parameter("data_left", leftType->Clone()));
		operands.push_back(new Identifier("data_left"));
	}

	parameters.push_back(new Parameter("data_right", rightType->Clone()));
	operands.push_back(new Identifier("data_right"));

	parameters.push_back(new Parameter("offsets", new BasicType(BasicType::BasicKind::Int64)));
	operands.push_back(new Identifier("offsets"));

	parameters.push_back(new Parameter("count", new BasicType(BasicType::BasicKind::Int64)));
	operands.push_back(new Identifier("count"));

	lvalues.push_back(new VariableDeclaration("indexes", new ListType(new BasicType(BasicType::BasicKind::Int64))));

	returnOperands.push_back(new Identifier("indexes"));
	returnTypes.push_back(new ListType(new BasicType(BasicType::BasicKind::Int64)));

	auto joinName = (isHashing) ? "hash_join" : "loop_join";
	auto joinCall = new CallExpression(new FunctionLiteral(new Identifier("GPU", joinName)), operands);
	auto joinStatement = new AssignStatement(lvalues, joinCall);
	auto returnStatement = new ReturnStatement(returnOperands);

	return new Function("join_" + std::to_string(m_index++), parameters, returnTypes, {joinStatement, returnStatement}, true);
}

Function *OutlineLibrary::GenerateMemberHashFunction(const Type *dataType)
{
	std::vector<Parameter *> parameters;

	std::vector<Operand *> operands;
	std::vector<LValue *> lvalues;

	std::vector<Operand *> returnOperands;
	std::vector<Type *> returnTypes;

	parameters.push_back(new Parameter("data", dataType->Clone()));
	operands.push_back(new Identifier("data"));

	lvalues.push_back(new VariableDeclaration("keys", dataType->Clone()));
	returnOperands.push_back(new Identifier("keys"));
	returnTypes.push_back(dataType->Clone());

	auto hashCall = new CallExpression(new FunctionLiteral(new Identifier("GPU", "hash_member_create")), operands);
	auto hashStatement = new AssignStatement(lvalues, hashCall);
	auto returnStatement = new ReturnStatement(returnOperands);

	return new Function("hash_member_create_" + std::to_string(m_index++), parameters, returnTypes, {hashStatement, returnStatement}, true);
}

Function *OutlineLibrary::GenerateMemberFunction(const Type *leftType, const Type *rightType)
{
	std::vector<Parameter *> parameters;

	std::vector<Operand *> operands;
	std::vector<LValue *> lvalues;

	std::vector<Operand *> returnOperands;
	std::vector<Type *> returnTypes;

	parameters.push_back(new Parameter("hash_keys", leftType->Clone()));
	operands.push_back(new Identifier("hash_keys"));

	parameters.push_back(new Parameter("data_right", rightType->Clone()));
	operands.push_back(new Identifier("data_right"));

	lvalues.push_back(new VariableDeclaration("matches", new BasicType(BasicType::BasicKind::Boolean)));
	returnOperands.push_back(new Identifier("matches"));
	returnTypes.push_back(new BasicType(BasicType::BasicKind::Boolean));

	auto memberCall = new CallExpression(new FunctionLiteral(new Identifier("GPU", "hash_member")), operands);
	auto memberStatement = new AssignStatement(lvalues, memberCall);
	auto returnStatement = new ReturnStatement(returnOperands);

	return new Function("hash_member_" + std::to_string(m_index++), parameters, returnTypes, {memberStatement, returnStatement}, true);
}

}
}
