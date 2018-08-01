#include "HorseIR/ShapeAnalysis.h"
#include "HorseIR/SymbolTable.h"
#include "HorseIR/Tree/Program.h"
#include "HorseIR/Tree/Method.h"
#include "HorseIR/Tree/Module.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Options.h"

#include "Interpreter/Interpreter.h"

#include "Runtime/Runtime.h"

//TODO: Move these somewhere
// #include "PTX/ArithmeticTest.h"
// #include "PTX/ComparisonTest.h"
// #include "PTX/ControlFlowTest.h"
// #include "PTX/DataTest.h"
// #include "PTX/LogicalTest.h"
// #include "PTX/ShiftTest.h"
// #include "PTX/SynchronizationTest.h"

// #include "PTX/AddTest.h"
// #include "PTX/BasicTest.h"
// #include "PTX/ConditionalTest.h"

int yyparse();

HorseIR::Program *program;

int main(int argc, const char *argv[])
{
	// Initialize the input arguments from the command line

	Utils::Options::Initialize(argc, argv);

	// Initialize the runtime environment and check that the machine is capable of running the query

	Runtime::Runtime runtime;
	runtime.Initialize();

	// Parse the input HorseIR program from stdin and generate an AST

	Utils::Logger::LogSection("Parsing input program");
	auto timeFrontend_start = Utils::Chrono::Start();

	yyparse();
	
	auto timeFrontend = Utils::Chrono::End(timeFrontend_start);

	if (Utils::Options::Present(Utils::Options::Opt_Dump_hir))
	{
		// Pretty print the input HorseIR program

		Utils::Logger::LogInfo("Input HorseIR program");
		Utils::Logger::LogInfo(program->ToString(), Utils::Logger::NoPrefix);
	}
	Utils::Logger::LogTiming("HorseIR frontend", timeFrontend);

	Utils::Logger::LogSection("Builtin symbol table");
	auto timeTypes_start = Utils::Chrono::Start();

	//TODO: Add builtin module
	auto m1 = new HorseIR::Method("load_table", {}, nullptr, {});
	auto m2 = new HorseIR::Method("column_value", {}, nullptr, {});
	auto m3 = new HorseIR::Method("enlist", {}, nullptr, {});
	auto m4 = new HorseIR::Method("table", {}, nullptr, {});

	auto m5 = new HorseIR::Method("geq", {}, nullptr, {});
	auto m6 = new HorseIR::Method("compress", {}, nullptr, {});
	auto m7 = new HorseIR::Method("not", {}, nullptr, {});
	auto m8 = new HorseIR::Method("mul", {}, nullptr, {});
	auto m9 = new HorseIR::Method("sum", {}, nullptr, {});

	auto builtinModule = new HorseIR::Module("Builtin", {m1, m2, m3, m4, m5, m6, m7, m8, m9});
	program->AddModule(builtinModule);

	HorseIR::SymbolTableBuilder symbolTable;
	symbolTable.Build(program);

	auto timeTypes = Utils::Chrono::End(timeTypes_start);

	if (Utils::Options::Present(Utils::Options::Opt_Dump_symtab))
	{
		// Dump the symbol table to stdout

		HorseIR::SymbolTableDumper dump;
		dump.Dump(program);
	}
	Utils::Logger::LogTiming("Symbol table", timeTypes);

	// Execute the HorseIR program in an interpeter, compiling GPU sections as needed

	Interpreter::Interpreter interpreter(runtime);
	interpreter.Execute(program);
}
