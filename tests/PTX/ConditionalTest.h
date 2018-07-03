#pragma once

#include "Test.h"

#include "CUDA/Buffer.h"
#include "CUDA/Kernel.h"
#include "CUDA/KernelInvocation.h"
#include "CUDA/Module.h"

#include "PTX/Module.h"
#include "PTX/StateSpace.h"
#include "PTX/Type.h"
#include "PTX/Functions/Function.h"
#include "PTX/Functions/DataFunction.h"
#include "PTX/Instructions/Arithmetic/AddInstruction.h"
#include "PTX/Instructions/Arithmetic/MultiplyWideInstruction.h"
#include "PTX/Instructions/Arithmetic/RemainderInstruction.h"
#include "PTX/Instructions/Comparison/SetPredicateInstruction.h"
#include "PTX/Instructions/ControlFlow/BranchInstruction.h"
#include "PTX/Instructions/ControlFlow/ReturnInstruction.h"
#include "PTX/Instructions/Data/ConvertAddressInstruction.h"
#include "PTX/Instructions/Data/LoadInstruction.h"
#include "PTX/Instructions/Data/MoveInstruction.h"
#include "PTX/Instructions/Data/StoreInstruction.h"
#include "PTX/Operands/Adapters/PointerAdapter.h"
#include "PTX/Operands/Address/Address.h"
#include "PTX/Operands/Address/RegisterAddress.h"
#include "PTX/Operands/Address/MemoryAddress.h"
#include "PTX/Operands/Value.h"
#include "PTX/Operands/Variables/Variable.h"
#include "PTX/Operands/Variables/IndexedRegister.h"
#include "PTX/Operands/Variables/Register.h"
#include "PTX/Statements/Label.h"

namespace Test
{

class ConditionalTest : public Test
{
public:
	ConditionalTest() {}
	
	void Execute()
	{
		PTX::Module module;

		module.SetVersion(6, 1);
		module.SetDeviceTarget("sm_61");
		module.SetAddressSize(PTX::Bits::Bits64);

		PTX::DataFunction<PTX::VoidType(PTX::ParameterVariable<PTX::Pointer64Type<PTX::UInt64Type>>)> *function = new PTX::DataFunction<PTX::VoidType(PTX::ParameterVariable<PTX::Pointer64Type<PTX::UInt64Type>>)>();
		function->SetName("ConditionalTest");
		function->SetEntry(true);
		function->SetLinkDirective(PTX::Declaration::LinkDirective::Visible);

		PTX::Pointer64Declaration<PTX::UInt64Type> *parameterDeclaration = new PTX::Pointer64Declaration<PTX::UInt64Type>("ConditionalTest_0");
		function->SetParameters(parameterDeclaration);

		PTX::SpecialRegisterDeclaration<PTX::Vector4Type<PTX::UInt32Type>> *srtid = new PTX::SpecialRegisterDeclaration<PTX::Vector4Type<PTX::UInt32Type>>("%tid");
		PTX::RegisterDeclaration<PTX::UInt32Type> *r32 = new PTX::RegisterDeclaration<PTX::UInt32Type>("%r", 3);
		PTX::RegisterDeclaration<PTX::UInt64Type> *r64 = new PTX::RegisterDeclaration<PTX::UInt64Type>("%rd", 4);
		PTX::RegisterDeclaration<PTX::PredicateType> *pDeclaration = new PTX::RegisterDeclaration<PTX::PredicateType>("p");

		const PTX::ParameterVariable<PTX::Pointer64Type<PTX::UInt64Type>> *parameter = parameterDeclaration->GetVariable("ConditionalTest_0");

		PTX::Register<PTX::UInt32Type> *tidx = new PTX::IndexedRegister4<PTX::UInt32Type>(srtid->GetVariable("%tid"), PTX::VectorElement::X);
																		       
		const PTX::Register<PTX::UInt32Type> *r0 = r32->GetVariable("%r", 0);
		const PTX::Register<PTX::UInt32Type> *r1 = r32->GetVariable("%r", 1);
		const PTX::Register<PTX::UInt32Type> *r2 = r32->GetVariable("%r", 2);

		const PTX::Register<PTX::UInt64Type> *rd0 = r64->GetVariable("%rd", 0);
		const PTX::Register<PTX::UInt64Type> *rd1 = r64->GetVariable("%rd", 1);
		const PTX::Register<PTX::UInt64Type> *rd2 = r64->GetVariable("%rd", 2);
		const PTX::Register<PTX::UInt64Type> *rd3 = r64->GetVariable("%rd", 3);

		PTX::Register<PTX::Pointer64Type<PTX::UInt64Type>> *rd0_ptr = new PTX::Pointer64RegisterAdapter<PTX::UInt64Type>(rd0);
		PTX::Register<PTX::Pointer64Type<PTX::UInt64Type, PTX::GlobalSpace>> *rd1_ptr = new PTX::Pointer64RegisterAdapter<PTX::UInt64Type, PTX::GlobalSpace>(rd1);

		const PTX::Register<PTX::PredicateType> *p = pDeclaration->GetVariable("p");

		function->AddStatement(r32);
		function->AddStatement(r64); 
		function->AddStatement(pDeclaration);

		function->AddStatement(new PTX::Load64Instruction<PTX::Pointer64Type<PTX::UInt64Type>, PTX::ParameterSpace>(rd0_ptr, new PTX::MemoryAddress64<PTX::Pointer64Type<PTX::UInt64Type>, PTX::ParameterSpace>(parameter)));
		function->AddStatement(new PTX::ConvertToAddress64Instruction<PTX::UInt64Type, PTX::GlobalSpace>(rd1_ptr, new PTX::RegisterAddress64<PTX::UInt64Type>(rd0_ptr)));
		function->AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(r0, tidx));
		function->AddStatement(new PTX::MultiplyWideInstruction<PTX::UInt64Type, PTX::UInt32Type>(rd2, r0, new PTX::UInt32Value(4)));
		function->AddStatement(new PTX::AddInstruction<PTX::UInt64Type>(rd3, rd1, rd2));

		function->AddStatement(new PTX::RemainderInstruction<PTX::UInt32Type>(r1, r0, new PTX::UInt32Value(2)));
		function->AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(p, r1, new PTX::UInt32Value(0), PTX::UInt32Type::ComparisonOperator::NotEqual));

		PTX::Label *labelFalse = new PTX::Label("false");
		PTX::Label *labelEnd = new PTX::Label("end");

		PTX::BranchInstruction *falseBranch = new PTX::BranchInstruction(labelFalse);
                falseBranch->SetPredicate(p);
		function->AddStatement(falseBranch);
		function->AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(r2, r0, new PTX::UInt32Value(1)));
		function->AddStatement(new PTX::BranchInstruction(labelEnd));
		function->AddStatement(labelFalse);
		function->AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(r2, r0, new PTX::UInt32Value(2)));
		function->AddStatement(labelEnd);

		PTX::Register<PTX::Pointer64Type<PTX::UInt32Type, PTX::GlobalSpace>> *rd3_ptr = new PTX::Pointer64RegisterAdapter<PTX::UInt32Type, PTX::GlobalSpace>(rd3);

		function->AddStatement(new PTX::Store64Instruction<PTX::UInt32Type, PTX::GlobalSpace>(new PTX::RegisterAddress64<PTX::UInt32Type, PTX::GlobalSpace>(rd3_ptr), r2));
		function->AddStatement(new PTX::ReturnInstruction());

		module.AddDeclaration(function);
		std::string ptx = module.ToString();
		std::cout << ptx;

		CUDA::Module cModule(ptx);
		CUDA::Kernel kernel(function->GetName(), 1, cModule);

		size_t size = sizeof(int) * 1024;
		int *data = (int *)malloc(size);
		std::memset(data, 0, size);

		CUDA::Buffer buffer(data, size);
		buffer.AllocateOnGPU();
		buffer.TransferToGPU();

		CUDA::KernelInvocation invocation(kernel);
		invocation.SetBlockShape(1024, 1, 1);
		invocation.SetGridShape(1, 1, 1);
		invocation.SetParam(0, buffer);
		invocation.Launch();

		buffer.TransferToCPU();

		for (int i = 0; i < 1024; ++i)
		{
			if ((i % 2 == 0 && data[i] == i + 1) || (i % 2 == 1 && data[i] == i + 2))
			{
				continue;
			}

			std::cerr << "[ERROR] Result incorrect at index " << i << " [" << data[i] << " != " << ((i % 2 == 0) ? i + 1 : i + 2) << "]" << std::endl;
			return;
		}
		std::cerr << "[INFO] Kernel execution successful" << std::endl;
	}
};

}
