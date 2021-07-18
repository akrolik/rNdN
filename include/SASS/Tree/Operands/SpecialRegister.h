#pragma once

#include "SASS/Tree/Operands/Operand.h"

namespace SASS {

class SpecialRegister : public Operand
{
public:
	enum class Kind {
		SR_LANEID                    = 0,
		SR_CLOCK                     = 1,
		SR_VIRTCFG                   = 2,
		SR_VIRTID                    = 3,
		SR_PM0                       = 4,
		SR_PM1                       = 5,
		SR_PM2                       = 6,
		SR_PM3                       = 7,
		SR_PM4                       = 8,
		SR_PM5                       = 9,

		SR_PM6                       = 10,
		SR_PM7                       = 11,
		SR_UNKNOWN_12                = 12,
		SR_UNKNOWN_13                = 13,
		SR_UNKNOWN_14                = 14,
		SR_UNKNOWN_15                = 15,
		SR_PRIM_TYPE                 = 16,
		SR_INVOCATION_ID             = 17,
		SR_Y_DIRECTION               = 18,
		SR_THREAD_KILL               = 19,

		SM_SHADER_TYPE               = 20,
		SR_DIRECTCBEWRITEADDRESSLOW  = 21,
		SR_DIRECTCBEWRITEADDRESSHIGH = 22,
		SR_DIRECTCBEWRITEENABLED     = 23,
		SR_MACHINE_ID_0              = 24,
		SR_MACHINE_ID_1              = 25,
		SR_MACHINE_ID_2              = 26,
		SR_MACHINE_ID_3              = 27,
		SR_AFFINITY                  = 28,
		SR_INVOCATION_INFO           = 29,

		SR_WSCALEFACTOR_XY           = 30,
		SR_WSCALEFACTOR_Z            = 31,
		SR_TID                       = 32,
		SR_TID_X                     = 33,
		SR_TID_Y                     = 34,
		SR_TID_Z                     = 35,
		SR_CTA_PARAM                 = 36,
		SR_CTAID_X                   = 37,
		SR_CTAID_Y                   = 38,
		SR_CTAID_Z                   = 39,

		SR_NTID                      = 40,
		SR_NTID_X                    = 41,
		SR_NTID_Y                    = 42,
		SR_NTID_Z                    = 43,
		SR_GRIDPARAM                 = 44,
		SR_NCTAID_X                  = 45,
		SR_NCTAID_Y                  = 46,
		SR_NCTAID_Z                  = 47,
		SR_SWINLO                    = 48,
		SR_SWINSZ                    = 49,

		SR_SMEMSZ                    = 50,
		SR_SMEMBANKS                 = 51,
		SR_LWINLO                    = 52,
		SR_LWINSZ                    = 53,
		SR_LMEMLOSZ                  = 54,
		SR_LMEMHIOFF                 = 55,
		SR_EQMASK                    = 56,
		SR_LTMASK                    = 57,
		SR_LEMASK                    = 58,
		SR_GTMASK                    = 59,

		SR_GEMASK                    = 60,
		SR_REGALLOC                  = 61,
		SR_CTXADDR                   = 62,
		SR_UNKNOWN_63                = 63,
		SR_GLOBALERRORSTATUS         = 64,
		SR_UNKNOWN_65                = 65,
		SR_WARPERRORSTATUS           = 66,
		SR_WARPERRORSTATUSCLEAR      = 67,
		SR_UNKNOWN_68                = 68,
		SR_UNKNOWN_69                = 69,

		SR_UNKNOWN_70                = 70,
		SR_UNKNOWN_71                = 71,
		SR_PM_HI0                    = 72,
		SR_PM_HI1                    = 73,
		SR_PM_HI2                    = 74,
		SR_PM_HI3                    = 75,
		SR_PM_HI4                    = 76,
		SR_PM_HI5                    = 77,
		SR_PM_HI6                    = 78,
		SR_PM_HI7                    = 79,

		SR_CLOCKLO                   = 80,
		SR_CLOCKHI                   = 81,
		SR_GLOBALTIMERLO             = 82,
		SR_GLOBALTIMERHI             = 83,
		SR_UNKNOWN_84                = 84,
		SR_UNKNOWN_85                = 85,
		SR_UNKNOWN_86                = 86,
		SR_UNKNOWN_87                = 87,
		SR_UNKNOWN_88                = 88,
		SR_UNKNOWN_89                = 89,

		SR_UNKNOWN_90                = 90,
		SR_UNKNOWN_91                = 91,
		SR_UNKNOWN_92                = 92,
		SR_UNKNOWN_93                = 93,
		SR_UNKNOWN_94                = 94,
		SR_UNKNOWN_95                = 95,
		SR_HWTASKID                  = 96,
		SR_CIRCULARQUEUEENTRYINDEX   = 97,
		SR_CIRCULARQUEUEENTRYADDRESSLOW  = 98,
		SR_CIRCULARQUEUEENTRYADDRESSHIGH = 99
	};

	static std::string KindString(Kind kind)
	{
#define STR(x) #x
#define CASE(x) case Kind::x: return STR(x)

		switch (kind)
		{
			CASE(SR_LANEID);
			CASE(SR_CLOCK);
			CASE(SR_VIRTCFG);
			CASE(SR_VIRTID);
			CASE(SR_PM0);
			CASE(SR_PM1);
			CASE(SR_PM2);
			CASE(SR_PM3);
			CASE(SR_PM4);
			CASE(SR_PM5);

			CASE(SR_PM6);
			CASE(SR_PM7);
			CASE(SR_UNKNOWN_12);
			CASE(SR_UNKNOWN_13);
			CASE(SR_UNKNOWN_14);
			CASE(SR_UNKNOWN_15);
			CASE(SR_PRIM_TYPE);
			CASE(SR_INVOCATION_ID);
			CASE(SR_Y_DIRECTION);
			CASE(SR_THREAD_KILL);

			CASE(SM_SHADER_TYPE);
			CASE(SR_DIRECTCBEWRITEADDRESSLOW);
			CASE(SR_DIRECTCBEWRITEADDRESSHIGH);
			CASE(SR_DIRECTCBEWRITEENABLED);
			CASE(SR_MACHINE_ID_0);
			CASE(SR_MACHINE_ID_1);
			CASE(SR_MACHINE_ID_2);
			CASE(SR_MACHINE_ID_3);
			CASE(SR_AFFINITY);
			CASE(SR_INVOCATION_INFO);

			CASE(SR_WSCALEFACTOR_XY);
			CASE(SR_WSCALEFACTOR_Z);
			CASE(SR_TID);
			CASE(SR_TID_X);
			CASE(SR_TID_Y);
			CASE(SR_TID_Z);
			CASE(SR_CTA_PARAM);
			CASE(SR_CTAID_X);
			CASE(SR_CTAID_Y);
			CASE(SR_CTAID_Z);

			CASE(SR_NTID);
			CASE(SR_NTID_X);
			CASE(SR_NTID_Y);
			CASE(SR_NTID_Z);
			CASE(SR_GRIDPARAM);
			CASE(SR_NCTAID_X);
			CASE(SR_NCTAID_Y);
			CASE(SR_NCTAID_Z);
			CASE(SR_SWINLO);
			CASE(SR_SWINSZ);

			CASE(SR_SMEMSZ);
			CASE(SR_SMEMBANKS);
			CASE(SR_LWINLO);
			CASE(SR_LWINSZ);
			CASE(SR_LMEMLOSZ);
			CASE(SR_LMEMHIOFF);
			CASE(SR_EQMASK);
			CASE(SR_LTMASK);
			CASE(SR_LEMASK);
			CASE(SR_GTMASK);

			CASE(SR_GEMASK);
			CASE(SR_REGALLOC);
			CASE(SR_CTXADDR);
			CASE(SR_UNKNOWN_63);
			CASE(SR_GLOBALERRORSTATUS);
			CASE(SR_UNKNOWN_65);
			CASE(SR_WARPERRORSTATUS);
			CASE(SR_WARPERRORSTATUSCLEAR);
			CASE(SR_UNKNOWN_68);
			CASE(SR_UNKNOWN_69);

			CASE(SR_UNKNOWN_70);
			CASE(SR_UNKNOWN_71);
			CASE(SR_PM_HI0);
			CASE(SR_PM_HI1);
			CASE(SR_PM_HI2);
			CASE(SR_PM_HI3);
			CASE(SR_PM_HI4);
			CASE(SR_PM_HI5);
			CASE(SR_PM_HI6);
			CASE(SR_PM_HI7);

			CASE(SR_CLOCKLO);
			CASE(SR_CLOCKHI);
			CASE(SR_GLOBALTIMERLO);
			CASE(SR_GLOBALTIMERHI);
			CASE(SR_UNKNOWN_84);
			CASE(SR_UNKNOWN_85);
			CASE(SR_UNKNOWN_86);
			CASE(SR_UNKNOWN_87);
			CASE(SR_UNKNOWN_88);
			CASE(SR_UNKNOWN_89);

			CASE(SR_UNKNOWN_90);
			CASE(SR_UNKNOWN_91);
			CASE(SR_UNKNOWN_92);
			CASE(SR_UNKNOWN_93);
			CASE(SR_UNKNOWN_94);
			CASE(SR_UNKNOWN_95);
			CASE(SR_HWTASKID);
			CASE(SR_CIRCULARQUEUEENTRYINDEX);
			CASE(SR_CIRCULARQUEUEENTRYADDRESSLOW);
			CASE(SR_CIRCULARQUEUEENTRYADDRESSHIGH);
		}
		return "<unknown>";
	}

	SpecialRegister(Kind kind) : Operand(Operand::Kind::SpecialRegister), m_kind(kind) {}

	// Properties

	Kind GetKind() const { return m_kind; }
	void SetKind(Kind kind) { m_kind = kind; }

	// Formatting

	std::string ToString() const override
	{
		return KindString(m_kind);
	}

	// Binary

	std::uint64_t ToBinary() const override
	{
		return static_cast<std::underlying_type<Kind>::type>(m_kind);
	}

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }
	
private:
	Kind m_kind;
};

}
