#pragma once

class SaturateModifier
{
public:
	SaturateModifier() {}
	SaturateModifier(bool saturate) : m_saturate(saturate) {}

	bool GetSaturate() const { return m_saturate; }
	void SetSaturate(bool saturate) { m_saturate = saturate; }

protected:
	bool m_saturate = false;
};
