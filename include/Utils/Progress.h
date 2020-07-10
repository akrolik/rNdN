#include <iostream>
#include <string>

namespace Utils {

class Progress
{
public:
	static Progress Start(const std::string& name, bool print)
	{
		Progress p(name, print);
		p.Update(0, 0);
		return p;
	}

	void Update(unsigned int complete, unsigned int total)
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

	void Complete()
	{
		if (m_print)
		{
			std::cout << std::endl;
		}
	}

	static unsigned int GetWidth() { return s_width; }
	static void SetWidth(unsigned int width) { s_width = width; }

private:
	Progress(const std::string& name, bool print) : m_name(name), m_print(print) {}

	std::string m_name;
	bool m_print = false;
	static unsigned int s_width;
};

unsigned int Progress::s_width = 30;

}
