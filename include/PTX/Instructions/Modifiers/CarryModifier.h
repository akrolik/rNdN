#pragma once

namespace PTX {

template<class T, bool force = false, typename Enable = void>
class CarryModifier
{
};

template<class T, bool force>
class CarryModifier<T, force, std::enable_if_t<force || T::CarryModifier>>
{
public:
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
