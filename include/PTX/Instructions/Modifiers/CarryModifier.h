#pragma once

namespace PTX {

template<class T, bool force = false, typename Enable = void>
class CarryModifier
{
public:
	constexpr static bool Enabled = false;
};

template<class T, bool force>
class CarryModifier<T, force,
      std::enable_if_t<TypeEnforcer<T, Int32Type, Int64Type, UInt32Type, UInt64Type>::value || force>>
{
public:
	constexpr static bool Enabled = true;

	CarryModifier() {}
	CarryModifier(bool carryIn, bool carryOut) : m_carryIn(carryIn), m_carryOut(carryOut) {}

	bool GetCarryIn() const { return m_carryIn; }
	void SetCarryIn(bool carryIn) { m_carryIn = carryIn; }

	bool GetCarryOut() const { return m_carryOut; }
	void SetCarryOut(bool carryOut) { m_carryOut = carryOut; }
	
	std::string OpCodeModifier() const
	{
		std::string ret;
		if (m_carryIn)
		{
			ret += "c";
		}
		if (m_carryOut)
		{
			ret += ".cc";
		}
		return ret;
	}

	bool IsActive() const
	{
		return m_carryIn || m_carryOut;
	}

protected:
	bool m_carryIn = false;
	bool m_carryOut = false;
};

}
