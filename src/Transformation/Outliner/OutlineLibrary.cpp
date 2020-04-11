#include "Transformation/Outliner/OutlineLibrary.h"

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

HorseIR::CallExpression *OutlineLibrary::Outline(const HorseIR::BuiltinFunction *function, const std::vector<HorseIR::Operand *>& arguments)
{
	switch (function->GetPrimitive())
	{
		// case HorseIR::BuiltinFunction::Primitive::Unique:
		case HorseIR::BuiltinFunction::Primitive::Group:
		{
			auto dataType = arguments.at(0)->GetType();
			auto orderLiteral = new HorseIR::BooleanLiteral({true});

			auto initFunction = GenerateInitFunction(dataType, orderLiteral);
			auto sortFunction = GenerateSortFunction(dataType, orderLiteral);
			auto groupFunction = GenerateGroupFunction(dataType);

			m_functions.push_back(initFunction);
			m_functions.push_back(sortFunction);
			m_functions.push_back(groupFunction);

			// Build the list of arguments for the library function call

			std::vector<HorseIR::Operand *> operands;
			operands.push_back(new HorseIR::FunctionLiteral(new HorseIR::Identifier(initFunction->GetName())));
			operands.push_back(new HorseIR::FunctionLiteral(new HorseIR::Identifier(sortFunction->GetName())));
			operands.push_back(new HorseIR::FunctionLiteral(new HorseIR::Identifier(groupFunction->GetName())));
			operands.push_back(arguments.at(0)->Clone());

			// Build the library call

			return new HorseIR::CallExpression(new HorseIR::FunctionLiteral(new HorseIR::Identifier("GPU", "group_lib")), operands);
		}
		case HorseIR::BuiltinFunction::Primitive::Order:
		{
			auto dataType = arguments.at(0)->GetType();
			auto orderLiteral = dynamic_cast<const HorseIR::BooleanLiteral *>(arguments.at(1));

			auto initFunction = GenerateInitFunction(dataType, orderLiteral);
			auto sortFunction = GenerateSortFunction(dataType, orderLiteral);

			m_functions.push_back(initFunction);
			m_functions.push_back(sortFunction);

			// Build the list of arguments for the library function call

			std::vector<HorseIR::Operand *> operands;
			operands.push_back(new HorseIR::FunctionLiteral(new HorseIR::Identifier(initFunction->GetName())));
			operands.push_back(new HorseIR::FunctionLiteral(new HorseIR::Identifier(sortFunction->GetName())));
			operands.push_back(arguments.at(0)->Clone());
			operands.push_back(arguments.at(1)->Clone());

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

			auto countFunction = GenerateJoinCountFunction(functions, leftType, rightType);
			auto joinFunction = GenerateJoinFunction(functions, leftType, rightType);

			m_functions.push_back(countFunction);
			m_functions.push_back(joinFunction);

			// Build the list of arguments for the library function call

			std::vector<HorseIR::Operand *> operands;
			operands.push_back(new HorseIR::FunctionLiteral(new HorseIR::Identifier(countFunction->GetName())));
			operands.push_back(new HorseIR::FunctionLiteral(new HorseIR::Identifier(joinFunction->GetName())));

			operands.push_back(arguments.at(0)->Clone());
			operands.push_back(arguments.at(0)->Clone());

			// Build the library call

			return new HorseIR::CallExpression(new HorseIR::FunctionLiteral(new HorseIR::Identifier("GPU", "join_lib")), operands);
		}
		default:
		{
			Utils::Logger::LogError("GPU library outliner does not support builtin function '" + function->GetName() + "'");
		}
	}
}

HorseIR::Function *OutlineLibrary::GenerateInitFunction(const HorseIR::Type *dataType, const HorseIR::BooleanLiteral *orders)
{
	std::vector<HorseIR::Parameter *> parameters;

	std::vector<HorseIR::Operand *> operands;
	std::vector<HorseIR::LValue *> lvalues;

	std::vector<HorseIR::Operand *> returnOperands;
	std::vector<HorseIR::Type *> returnTypes;

	parameters.push_back(new HorseIR::Parameter("data", dataType->Clone()));
	operands.push_back(new HorseIR::Identifier("data"));

	parameters.push_back(new HorseIR::Parameter("orders", new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Boolean)));
	if (orders == nullptr)
	{
		operands.push_back(new HorseIR::Identifier("orders"));
	}
	else
	{
		operands.push_back(orders->Clone());
	}

	lvalues.push_back(new HorseIR::VariableDeclaration("index", new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64)));
	returnOperands.push_back(new HorseIR::Identifier("index"));
	returnTypes.push_back(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64));

	lvalues.push_back(new HorseIR::VariableDeclaration("data_out", dataType->Clone()));
	returnOperands.push_back(new HorseIR::Identifier("data_out"));
	returnTypes.push_back(dataType->Clone());

	auto initCall = new HorseIR::CallExpression(new HorseIR::FunctionLiteral(new HorseIR::Identifier("GPU", "order_init")), operands);
	auto initStatement = new HorseIR::AssignStatement(lvalues, initCall);
	auto returnStatement = new HorseIR::ReturnStatement(returnOperands);

	return new HorseIR::Function("order_init_" + std::to_string(m_index++), parameters, returnTypes, {initStatement, returnStatement}, true);
}

HorseIR::Function *OutlineLibrary::GenerateSortFunction(const HorseIR::Type *dataType, const HorseIR::BooleanLiteral *orders)
{
	std::vector<HorseIR::Parameter *> parameters;
	std::vector<HorseIR::Operand *> operands;

	parameters.push_back(new HorseIR::Parameter("index", new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64)));
	operands.push_back(new HorseIR::Identifier("index"));

	parameters.push_back(new HorseIR::Parameter("data", dataType->Clone()));
	operands.push_back(new HorseIR::Identifier("data"));

	parameters.push_back(new HorseIR::Parameter("orders", new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Boolean)));
	if (orders == nullptr)
	{
		operands.push_back(new HorseIR::Identifier("orders"));
	}
	else
	{
		operands.push_back(orders->Clone());
	}

	auto sortCall = new HorseIR::CallExpression(new HorseIR::FunctionLiteral(new HorseIR::Identifier("GPU", "order")), operands);
	auto sortStatement = new HorseIR::ExpressionStatement(sortCall);
	
	return new HorseIR::Function("order_" + std::to_string(m_index++), parameters, {}, {sortStatement}, true);
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

HorseIR::Function *OutlineLibrary::GenerateJoinCountFunction(std::vector<const HorseIR::Operand *>& functions, const HorseIR::Type *leftType, const HorseIR::Type *rightType)
{
	std::vector<HorseIR::Parameter *> parameters;

	std::vector<HorseIR::Operand *> operands;
	std::vector<HorseIR::LValue *> lvalues;

	std::vector<HorseIR::Operand *> returnOperands;
	std::vector<HorseIR::Type *> returnTypes;

	for (const auto argument : functions)
	{
		operands.push_back(argument->Clone());
	}

	parameters.push_back(new HorseIR::Parameter("data_left", leftType->Clone()));
	operands.push_back(new HorseIR::Identifier("data_left"));

	parameters.push_back(new HorseIR::Parameter("data_right", rightType->Clone()));
	operands.push_back(new HorseIR::Identifier("data_right"));

	lvalues.push_back(new HorseIR::VariableDeclaration("count", new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64)));

	returnOperands.push_back(new HorseIR::Identifier("count"));
	returnTypes.push_back(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64));

	auto joinCall = new HorseIR::CallExpression(new HorseIR::FunctionLiteral(new HorseIR::Identifier("GPU", "join_count")), operands);
	auto joinStatement = new HorseIR::AssignStatement(lvalues, joinCall);
	auto returnStatement = new HorseIR::ReturnStatement(returnOperands);

	return new HorseIR::Function("join_count_" + std::to_string(m_index++), parameters, returnTypes, {joinStatement, returnStatement}, true);
}

HorseIR::Function *OutlineLibrary::GenerateJoinFunction(std::vector<const HorseIR::Operand *>& functions, const HorseIR::Type *leftType, const HorseIR::Type *rightType)
{
	std::vector<HorseIR::Parameter *> parameters;

	std::vector<HorseIR::Operand *> operands;
	std::vector<HorseIR::LValue *> lvalues;

	std::vector<HorseIR::Operand *> returnOperands;
	std::vector<HorseIR::Type *> returnTypes;

	for (const auto argument : functions)
	{
		operands.push_back(argument->Clone());
	}

	parameters.push_back(new HorseIR::Parameter("data_left", leftType->Clone()));
	operands.push_back(new HorseIR::Identifier("data_left"));

	parameters.push_back(new HorseIR::Parameter("data_right", rightType->Clone()));
	operands.push_back(new HorseIR::Identifier("data_right"));

	parameters.push_back(new HorseIR::Parameter("count", new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64)));
	operands.push_back(new HorseIR::Identifier("count"));

	lvalues.push_back(new HorseIR::VariableDeclaration("indexes", new HorseIR::ListType(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64))));

	returnOperands.push_back(new HorseIR::Identifier("indexes"));
	returnTypes.push_back(new HorseIR::ListType(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64)));

	auto joinCall = new HorseIR::CallExpression(new HorseIR::FunctionLiteral(new HorseIR::Identifier("GPU", "join")), operands);
	auto joinStatement = new HorseIR::AssignStatement(lvalues, joinCall);
	auto returnStatement = new HorseIR::ReturnStatement(returnOperands);

	return new HorseIR::Function("join_" + std::to_string(m_index++), parameters, returnTypes, {joinStatement, returnStatement}, true);

}

}
