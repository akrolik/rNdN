#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"

#include "PTX/Tree/Type.h"
#include "PTX/Tree/Operands/Constants/Value.h"
#include "PTX/Tree/Operands/Variables/Registers/Register.h"

namespace PTX {

DispatchInterface(ReduxInstruction)

template<class T>
class ReduxInstructionBase : DispatchInherit(ReduxInstruction), public PredicatedInstruction
{
public:
	ReduxInstructionBase(Register<T> *destination, Register<T> *source, UInt32Value *memberMask)
		: m_destination(destination), m_source(source), m_memberMask(memberMask) {}

	// Analysis properties

	bool HasSideEffect() const override { return true; }

	// Properties

	const Register<T> *GetDestination() const { return m_destination; }
	Register<T> *GetDestination() { return m_destination; }
	void SetDestination(Register<T> *destination) { m_destination = destination; }

	const Register<T> *GetSource() const { return m_source; }
	Register<T> *GetSource() { return m_source; }
	void SetSource(Register<T> *source) { m_source = source; }

	const UInt32Value *GetThreads() const { return m_memberMask; }
	UInt32Value *GetThreads() { return m_memberMask; }
	void SetThreads(UInt32Value *memberMask) { m_memberMask = memberMask; }

	// Formatting

	std::vector<const Operand *> GetOperands() const override
	{
		return { m_destination, m_source, m_memberMask };
	}

	std::vector<Operand *> GetOperands() override
	{
		return { m_destination, m_source, m_memberMask };
	}

	// Visitors

	void Accept(InstructionVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:                
	DispatchMember_Type(T);

	Register<T> *m_destination = nullptr;
	Register<T> *m_source = nullptr;
	UInt32Value *m_memberMask = nullptr;
};

template<class T, bool Assert = true>
class ReduxInstruction : public ReduxInstructionBase<T>
{
public:
	REQUIRE_TYPE_PARAM(ReduxInstruction,
		REQUIRE_EXACT(T, UInt32Type, Int32Type, Bit32Type)
	);

	// Formatting

	enum class Operation {
		Add,
		Min,
		Max
	};

	static std::string OperationString(Operation operation)
	{
		switch (operation)
		{
			case Operation::Add:
				return ".add";
			case Operation::Min:
				return ".min";
			case Operation::Max:
				return ".max";
		}
		return ".<unknown>";
	}

	ReduxInstruction(Register<T> *destination, Register<T> *source, UInt32Value *memberMask, Operation operation)
		: ReduxInstructionBase<T>(destination, source, memberMask), m_operation(operation) {}

	// Properties

	Operation GetOperation() const { return m_operation; }
	void SetOperation(Operation operation) { m_operation = operation; }

	// Formatting

	static std::string Mnemonic() { return "redux.sync"; }

	std::string GetOpCode() const override
	{
		return Mnemonic() + OperationString(m_operation) + T::Name();
	}

protected:
	Operation m_operation;
};

template<>
class ReduxInstruction<Bit32Type> : public ReduxInstructionBase<Bit32Type>
{
public:
	enum class Operation {
		And,
		Or,
		Xor
	};

	static std::string OperationString(Operation operation)
	{
		switch (operation)
		{
			case Operation::And:
				return ".and";
			case Operation::Or:
				return ".or";
			case Operation::Xor:
				return ".xor";
		}
		return ".<unknown>";
	}

	ReduxInstruction(Register<Bit32Type> *destination, Register<Bit32Type> *source, UInt32Value *memberMask, Operation operation)
		: ReduxInstructionBase<Bit32Type>(destination, source, memberMask), m_operation(operation) {}

	// Properties

	Operation GetOperation() const { return m_operation; }
	void SetOperation(Operation operation) { m_operation = operation; }

	// Formatting

	static std::string Mnemonic() { return "redux.sync"; }

	std::string GetOpCode() const override
	{
		return Mnemonic() + OperationString(m_operation) + Bit32Type::Name();
	}

protected:
	Operation m_operation;
};

}
