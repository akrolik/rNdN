#pragma once

#include "SASS/Tree/Instructions/Volta/PredicatedInstruction.h"
#include "SASS/Tree/Instructions/Volta/BinaryUtils.h"

#include "SASS/Tree/Operands/Address.h"
#include "SASS/Tree/Operands/Register.h"

namespace SASS {
namespace Volta {

class REDInstruction : public PredicatedInstruction
{
public:
	enum Flags : std::uint64_t {
		None  = 0x0,
		E     = (1 << 0),
	};

	SASS_FLAGS_FRIEND()

	enum class Cache : std::uint64_t {
		None                 = 0x0,
		CONSTANT             = 0x4,
		CONSTANT_GPU         = 0xf,
		CONSTANT_PRIVATE     = 0x1,
		CONSTANT_CTA         = 0x2,
		CONSTANT_CTA_PRIVATE = 0x3,
		CONSTANT_SM          = 0x9,
		CONSTANT_SM_PRIVATE  = 0xb,
		CONSTANT_VC          = 0xd,
		CONSTANT_VC_PRIVATE  = 0xe,
		STRONG_SM            = 0x5,
		STRONG_SYS           = 0xa,
		STRONG_GPU           = 0x7,
		STRONG_GPU_PRIVATE   = 0x6,
		MMIO_GPU             = 0x8,
		MMIO_SYS             = 0xc
	};

	enum class Evict : std::uint64_t {
		None = 0x1,
		EF   = 0x0,
		EL   = 0x2,
		LU   = 0x3,
		EU   = 0x4,
		NA   = 0x5
	};

	enum class Type : std::uint64_t {
		U32 = 0x0,
		S32 = 0x1,
		U64 = 0x2,
		F32 = 0x3,
		F16 = 0x4,
		S64 = 0x5,
		F64 = 0x6
	};

	enum class Mode : std::uint64_t {
		ADD = 0x0,
		MIN = 0x1,
		MAX = 0x2,
		INC = 0x3,
		DEC = 0x4,
		AND = 0x5,
		OR  = 0x6,
		XOR = 0x7
	};

	REDInstruction(Address *destination, Register *source, Type type, Mode mode, Cache cache = Cache::None, Flags flags = Flags::None)
		: m_destination(destination), m_source(source), m_type(type), m_mode(mode), m_cache(cache), m_flags(flags) {}

	// Properties

	const Address *GetDestination() const { return m_destination; }
	Address *GetDestination() { return m_destination; }
	void SetDestination(Address *destination) { m_destination = destination; }

	const Register *GetSource() const { return m_source; }
	Register *GetSource() { return m_source; }
	void SetSource(Register *source) { m_source = source; }

	Type GetType() const { return m_type; }
	void SetType(Type type) { m_type = type; }

	Mode GetMode() const { return m_mode; }
	void SetMode(Mode mode) { m_mode = mode; }

	Cache GetCache() const { return m_cache; }
	void SetCache(Cache cache) { m_cache = cache; }

	Evict GetEvict() const { return m_evict; }
	void SetEvict(Evict evict) { m_evict = evict; }

	Flags GetFlags() const { return m_flags; }
	void SetFlags(Flags flags) { m_flags = flags; }

	// Operands
	
	std::vector<Operand *> GetDestinationOperands() const override
	{
		return { m_destination };
	}

	std::vector<Operand *> GetSourceOperands() const override
	{
		return { m_source };
	}

	// Formatting

	std::string OpCode() const override { return "RED"; }

	std::string OpModifiers() const override
	{
		std::string code;
		if (m_flags & Flags::E)
		{
			code += ".E";
		}
		switch (m_mode)
		{
			case Mode::ADD: code += ".ADD"; break;
			case Mode::MIN: code += ".MIN"; break;
			case Mode::MAX: code += ".MAX"; break;
			case Mode::INC: code += ".INC"; break;
			case Mode::DEC: code += ".DEC"; break;
			case Mode::AND: code += ".AND"; break;
			case Mode::OR:  code += ".OR"; break;
			case Mode::XOR: code += ".XOR"; break;
		}
		switch (m_evict)
		{
			case Evict::EF: code += ".EF"; break;
			case Evict::EL: code += ".EL"; break;
			case Evict::LU: code += ".LU"; break;
			case Evict::EU: code += ".EU"; break;
			case Evict::NA: code += ".NA"; break;
		}
		switch (m_type)
		{
			// case Type::U32: code += ".U32"; break;
			case Type::S32: code += ".S32"; break;
			case Type::U64: code += ".64"; break;
			case Type::F32: code += ".F32.FTZ.RN"; break;
			case Type::F16: code += ".F16x2.RN"; break;
			case Type::S64: code += ".S64"; break;
			case Type::F64: code += ".F64.RN"; break;
		}
		switch (m_cache)
		{
			case Cache::CONSTANT: code += ".CONSTANT"; break;
			case Cache::CONSTANT_GPU: code += ".CONSTANT.GPU"; break;
			case Cache::CONSTANT_PRIVATE: code += ".CONSTANT.PRIVATE"; break;
			case Cache::CONSTANT_CTA: code += ".CONSTANT.CTA"; break;
			case Cache::CONSTANT_CTA_PRIVATE: code += ".CONSTANT.CTA.PRIVATE"; break;
			case Cache::CONSTANT_SM: code += ".CONSTANT.SM"; break;
			case Cache::CONSTANT_SM_PRIVATE: code += ".CONSTANT.SM.PRIVATE"; break;
			case Cache::CONSTANT_VC: code += ".CONSTANT.VC"; break;
			case Cache::CONSTANT_VC_PRIVATE: code += ".CONSTANT.VC.PRIVATE"; break;
			case Cache::STRONG_SM: code += ".STRONG.SM"; break;
			case Cache::STRONG_SYS: code += ".STRONG.SYS"; break;
			case Cache::STRONG_GPU: code += ".STRONG.GPU"; break;
			case Cache::STRONG_GPU_PRIVATE: code += ".STRONG.GPU.PRIVATE"; break;
			case Cache::MMIO_GPU: code += ".MMIO.GPU"; break;
			case Cache::MMIO_SYS: code += ".MMIO.SYS"; break;
		}
		return code;
	}

	std::string Operands() const override
	{
		std::string code;

		// Destination
		code += m_destination->ToSizedString();
		code += ", ";

		// Source
		code += m_source->ToString();

		return code;
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return 0x98e;
	}

	std::uint64_t ToBinary() const override
	{
		auto code = PredicatedInstruction::ToBinary();

		// Destination
		code |= BinaryUtils::OperandRegister24(m_destination->GetBase());
		code |= BinaryUtils::OperandAddress40(m_destination->GetOffset());

		// Source
		code |= BinaryUtils::OperandRegister32(m_source);

		return code;
	}

	std::uint64_t ToBinaryHi() const override
	{
		auto code = PredicatedInstruction::ToBinaryHi();

		// Flags
		code |= BinaryUtils::FlagBit(m_flags & Flags::E, 8);

		// Type
		code |= BinaryUtils::Format(m_type, 9, 0xf);

		// Cache
		code |= BinaryUtils::Format(m_cache, 13, 0xf);

		// Evict
		code |= BinaryUtils::Format(m_evict, 20, 0x7);

		// Mode
		code |= BinaryUtils::Format(m_mode, 23, 0x7);

		return code;
	}

	// Hardware properties

	InstructionClass GetInstructionClass() const override { return InstructionClass::GlobalMemoryStore; }

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }

private:
	Address *m_destination = nullptr;
	Register *m_source = nullptr;

	Type m_type;
	Mode m_mode;
	Evict m_evict = Evict::None;
	Cache m_cache = Cache::None;
	Flags m_flags = Flags::None;
};

SASS_FLAGS_INLINE(REDInstruction)

}
}
