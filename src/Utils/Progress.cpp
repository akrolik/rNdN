#include "Utils/Progress.h"

#include <iomanip>
#include <iostream>

namespace Utils {

unsigned int Progress::s_width = 30;

Progress Progress::Start(const std::string& name, bool print)
{
	Progress p(name, print);
	p.Update(0, 0);
	return p;
}

void Progress::Update(unsigned int complete, unsigned int total)
{
	if (complete % 100 == 0 || complete == total)
	{
		if (m_print)
		{
			auto progress = (total > 0) ? float(complete) / total : 0;

			std::cout << "[INFO] " << m_name << ": [";
			for (auto i = 0u; i < s_width; ++i)
			{
				if (i < s_width * progress)
				{
					std::cout << "#";
				}
				else
				{
					std::cout << " ";
				}
			}
			std::cout << "] ";
			if (total > 0)
			{
				std::cout << complete << "/" << total << " ";
			}
			std::cout << std::fixed << std::setprecision(1) << "(" << progress * 100 << " %)\r" << std::flush;
		}
	}
}

void Progress::Complete()
{
	if (m_print)
	{
		std::cout << std::endl;
	}
}

}
