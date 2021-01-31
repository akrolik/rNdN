#pragma once

#include "Assembler/BinaryProgram.h"
#include "Assembler/ELFBinary.h"

namespace Assembler {

class ELFGenerator
{
public:
	enum class Attribute : char {
		EIATTR_REG_COUNT       = 0x2f,
		EIATTR_MAX_STACK_SIZE  = 0x23,
		EIATTR_MIN_STACK_SIZE  = 0x12,
		EIATTR_FRAME_SIZE      = 0x11,

		EIATTR_SW2393858_WAR            = 0x30,
		EIATTR_SW1850030_WAR            = 0x2a,
		EIATTR_PARAM_CBANK              = 0x0a,
		EIATTR_CBANK_PARAM_SIZE         = 0x19,
		EIATTR_KPARAM_INFO              = 0x17,
		EIATTR_MAXREG_COUNT             = 0x1b,
		EIATTR_S2RCTAID_INSTR_OFFSETS   = 0x1d,
		EIATTR_EXIT_INSTR_OFFSETS       = 0x1c,
		EIATTR_COOP_GROUP_INSTR_OFFSETS = 0x28,
		EIATTR_CTAIDZ_USED              = 0x04,
		EIATTR_MAX_THREADS              = 0x05,
		EIATTR_REQNTID                  = 0x10
	};

	enum class Type : char {
		EIFMT_NVAL  = 0x01,
		EIFMT_HVAL  = 0x03,
		EIFMT_SVAL  = 0x04,
	};

	ELFBinary *Generate(const BinaryProgram *program);

private:
	template<class T>
	std::vector<char> DecomposeShort(const T& value);
	template<class T>
	std::vector<char> DecomposeWord(const T& value);

	void AppendBytes(std::vector<char>& buffer, const std::vector<char>& bytes);
};

}
