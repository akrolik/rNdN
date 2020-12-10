#pragma once

namespace PTX {

class UniformModifier
{
public:
	UniformModifier(bool uniform = false) : m_uniform(uniform) {}

	// Properties

	bool GetUniform() const { return m_uniform; }
	void SetUniform(bool uniform) { m_uniform = uniform; }

	// Formatting

	std::string GetOpCodeModifier() const
	{
		if (m_uniform)
		{
			return ".uni";
		}
		return "";
	}

protected:
	bool m_uniform = false;
};

}
