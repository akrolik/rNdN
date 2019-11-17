#include "CUDA/ExternalModule.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <vector>

#include "Utils/Chrono.h"
#include "Utils/Logger.h"

namespace CUDA {

std::string ReplaceString(std::string subject, const std::string& search, const std::string& replace)
{
	size_t pos = 0;
	while ((pos = subject.find(search, pos)) != std::string::npos)
	{
		subject.replace(pos, search.length(), replace);
		pos += replace.length();
	}
	return subject;
}

void ExternalModule::GenerateBinary(const std::unique_ptr<Device>& device)
{
	auto timeGenerate_start = Utils::Chrono::Start("Generate external binary '" + m_name + "'");

	std::string command = "ptxas";

	command += " --gpu-name " + device->GetComputeCapability();
	command += " --compile-only";
	command += " --output-file " + m_name + ".cubin";
	command += " --input-as-string \"" + ReplaceString(m_code, "\"", "\\\"") + "\"";

	FILE *file = popen(command.c_str(), "r"); 
	if (file != NULL)
	{
		int c;
		while ((c = getc(file)) != EOF)
		{
			putchar(c);
		}

		fclose(file);
	}

	std::ifstream infile(m_name + ".cubin", std::ios::binary | std::ios::ate);
	std::streamsize size = infile.tellg();
	infile.seekg(0, std::ios::beg);

	std::vector<char> buffer(size);
	if (!infile.read(buffer.data(), size))
	{
		Utils::Logger::LogError("Unable to load binary for external module " + m_name);
	}

	m_binary = ::operator new(size);
	std::memcpy(m_binary, buffer.data(), size);
	m_binarySize = size;

	Utils::Chrono::End(timeGenerate_start);
}

}
