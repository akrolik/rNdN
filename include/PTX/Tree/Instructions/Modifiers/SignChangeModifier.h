#pragma once

namespace PTX {

template<class T, bool force = false, typename Enable = void>
class SignChangeModifier
{
public:
	constexpr static bool Enabled = false;
};

template<class T, bool force>
class SignChangeModifier<T, force,
      std::enable_if_t<REQUIRE_EXACT(T, Float16Type, Float16x2Type, Float32Type) || force>>
{
public:
	constexpr static bool Enabled = true;

	SignChangeModifier(bool signChange = false) : m_signChange(signChange) {}

	// Properties

	bool GetSignChange() const { return m_signChange; }
	void SetSignChange(bool signChange) { m_signChange = signChange; }

	// Formatting

	std::string GetOpCodeModifier() const
	{
		if (m_signChange)
		{
			return ".xorsign.abs";
		}
		return "";
	}

protected:
	bool m_signChange = false;
};

}

