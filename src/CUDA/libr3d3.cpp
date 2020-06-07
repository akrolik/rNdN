#include "CUDA/libr3d3.h"

namespace CUDA {

PTX::Program *libr3d3::CreateProgram(const std::unique_ptr<Device>& device)
{
	auto program = new PTX::Program();
	auto module = new PTX::Module();
	module->SetVersion(PTX::MAJOR_VERSION, PTX::MINOR_VERSION);
	module->SetDeviceTarget(device->GetComputeCapability());
	module->SetAddressSize(PTX::Bits::Bits64);

	CreateFunction_set<PTX::Bits::Bits64, PTX::Int8Type>(module, "i8");
	CreateFunction_set<PTX::Bits::Bits64, PTX::Int16Type>(module, "i16");
	CreateFunction_set<PTX::Bits::Bits64, PTX::Int32Type>(module, "i32");
	CreateFunction_set<PTX::Bits::Bits64, PTX::Int64Type>(module, "i64");

	CreateFunction_set<PTX::Bits::Bits64, PTX::Float32Type>(module, "f32");
	CreateFunction_set<PTX::Bits::Bits64, PTX::Float64Type>(module, "f64");

	program->AddModule(module);
	return program;
}

template<PTX::Bits B, class T>
void libr3d3::CreateFunction_set(PTX::Module *module, const std::string& typeName)
{
	// .visible .entry set_i32(
	// 	.param .u64 data,
	// 	.param .u32 value,
	// 	.param .u32 size,
	// )
	// {
	// 	.reg .pred      %p<2>;
	// 	.reg .b32       %r<7>;
	// 	.reg .b64       %rd<5>;

	// 	ld.param.u64    %rd1, [data];
	// 	ld.param.u32    %r2, [value];
	// 	ld.param.u32    %r3, [size];

	// 	mov.u32         %r4, %ctaid.x;
	// 	mov.u32         %r5, %ntid.x;
	// 	mov.u32         %r6, %tid.x;
	// 	mad.lo.u32      %r1, %r5, %r4, %r6;
	// 	setp.ge.u32     %p1, %r1, %r3;
	// 	@%p1 bra        BB0_2;

	// 	cvta.to.global.u64      %rd2, %rd1;
	// 	mul.wide.s32    %rd3, %r1, 4;
	// 	add.s64         %rd4, %rd2, %rd3;
	// 	st.global.s32   [%rd4], %r2;

	// BB0_2:
	// 	ret;
	// }

	// Setup kernel

	auto kernel = new PTX::FunctionDefinition<PTX::VoidType>();
	kernel->SetName("set_" + typeName);
	kernel->SetEntry(true);
	kernel->SetLinkDirective(PTX::Declaration::LinkDirective::Visible);

	// Initialize parameters

	auto dataDeclaration = new PTX::PointerDeclaration<B, T>("data");
	auto valueDeclaration = new PTX::ParameterDeclaration<T>("value");
	auto sizeDeclaration = new PTX::ParameterDeclaration<PTX::UInt32Type>("size");

	kernel->AddParameter(dataDeclaration);
	kernel->AddParameter(valueDeclaration);
	kernel->AddParameter(sizeDeclaration);
	
	auto data = dataDeclaration->GetVariable("data");
	auto value = valueDeclaration->GetVariable("value");
	auto size = sizeDeclaration->GetVariable("size");

	// Allocate registers

	auto addressRegisterDeclaration = new PTX::RegisterDeclaration<PTX::UIntType<B>>("rd_a", 4);
	kernel->AddStatement(addressRegisterDeclaration);

	auto addressRegister0 = addressRegisterDeclaration->GetVariable("rd_a", 0);
	auto addressRegister1 = addressRegisterDeclaration->GetVariable("rd_a", 1);
	auto addressRegister2 = addressRegisterDeclaration->GetVariable("rd_a", 2);
	auto addressRegister3 = addressRegisterDeclaration->GetVariable("rd_a", 3);

	auto valueRegisterDeclaration = new PTX::RegisterDeclaration<T>("v", 1);
	kernel->AddStatement(valueRegisterDeclaration);
	auto valueRegister = valueRegisterDeclaration->GetVariable("v", 0);

	auto indexRegisterDeclaration = new PTX::RegisterDeclaration<PTX::UInt32Type>("r_i", 5);
	kernel->AddStatement(indexRegisterDeclaration);

	auto indexRegister0 = indexRegisterDeclaration->GetVariable("r_i", 0);
	auto indexRegister1 = indexRegisterDeclaration->GetVariable("r_i", 1);
	auto indexRegister2 = indexRegisterDeclaration->GetVariable("r_i", 2);
	auto indexRegister3 = indexRegisterDeclaration->GetVariable("r_i", 3);
	auto indexRegister4 = indexRegisterDeclaration->GetVariable("r_i", 4);

	auto predicateRegisterDeclaration = new PTX::RegisterDeclaration<PTX::PredicateType>("p", 1);
	kernel->AddStatement(predicateRegisterDeclaration);
	auto predicateRegister = predicateRegisterDeclaration->GetVariable("p", 0);

	// Load parameters

	auto dataAddressBase = new PTX::MemoryAddress<B, PTX::PointerType<B, T>, PTX::ParameterSpace>(data);
	auto dataBase = new PTX::PointerRegisterAdapter<B, T>(addressRegister0);

	auto valueAddress = new PTX::MemoryAddress<B, T, PTX::ParameterSpace>(value);
	auto sizeAddress = new PTX::MemoryAddress<B, PTX::UInt32Type, PTX::ParameterSpace>(size);

	kernel->AddStatement(new PTX::LoadInstruction<B, PTX::PointerType<B, T>, PTX::ParameterSpace>(dataBase, dataAddressBase));
	kernel->AddStatement(new PTX::LoadInstruction<B, T, PTX::ParameterSpace>(valueRegister, valueAddress));
	kernel->AddStatement(new PTX::LoadInstruction<B, PTX::UInt32Type, PTX::ParameterSpace>(indexRegister0, sizeAddress));
	
	// Compute index

	auto indexed_tid = new PTX::IndexedRegister4<PTX::UInt32Type>(PTX::SpecialRegisterDeclaration_tid->GetVariable("%tid"), PTX::VectorElement::X);
	auto indexed_ntid = new PTX::IndexedRegister4<PTX::UInt32Type>(PTX::SpecialRegisterDeclaration_ntid->GetVariable("%ntid"), PTX::VectorElement::X);
	auto indexed_ctaid = new PTX::IndexedRegister4<PTX::UInt32Type>(PTX::SpecialRegisterDeclaration_ctaid->GetVariable("%ctaid"), PTX::VectorElement::X);

	kernel->AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(indexRegister1, indexed_tid));
	kernel->AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(indexRegister2, indexed_ntid));
	kernel->AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(indexRegister3, indexed_ctaid));

	kernel->AddStatement(new PTX::MADInstruction<PTX::UInt32Type>(
		indexRegister4, indexRegister2, indexRegister3, indexRegister1, PTX::HalfModifier<PTX::UInt32Type>::Half::Lower
	));

	// Bounds check

	auto endLabel = new PTX::Label("END");
	kernel->AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(
		predicateRegister, indexRegister4, indexRegister0, PTX::UInt32Type::ComparisonOperator::GreaterEqual
	));
	kernel->AddStatement(new PTX::BranchInstruction(endLabel, predicateRegister));

	// Store result at index

	auto globalBase = new PTX::PointerRegisterAdapter<B, T, PTX::GlobalSpace>(addressRegister1);
	auto genericAddress = new PTX::RegisterAddress<B, T>(dataBase);

	kernel->AddStatement(new PTX::ConvertToAddressInstruction<B, T, PTX::GlobalSpace>(globalBase, genericAddress));

	if constexpr(B == PTX::Bits::Bits32)
	{
		kernel->AddStatement(new PTX::ShiftLeftInstruction<PTX::Bit32Type>(
			new PTX::Bit32RegisterAdapter<PTX::UIntType>(addressRegister2),
			new PTX::Bit32Adapter<PTX::UIntType>(indexRegister4),
			new PTX::UInt32Value(std::log2(PTX::BitSize<T::TypeBits>::NumBytes))
		));
	}
	else
	{
		kernel->AddStatement(new PTX::MultiplyWideInstruction<PTX::UInt32Type>(
			addressRegister2, indexRegister4, new PTX::UInt32Value(PTX::BitSize<T::TypeBits>::NumBytes)
		));
	}

	kernel->AddStatement(new PTX::AddInstruction<PTX::UIntType<B>>(addressRegister3, addressRegister1, addressRegister2));

	auto dataAddressIndexed = new PTX::RegisterAddress<B, T, PTX::GlobalSpace>(new PTX::PointerRegisterAdapter<B, T, PTX::GlobalSpace>(addressRegister3));
	kernel->AddStatement(new PTX::StoreInstruction<B, T, PTX::GlobalSpace>(dataAddressIndexed, valueRegister));

	// End kernel with return

	kernel->AddStatement(endLabel);
	kernel->AddStatement(new PTX::ReturnInstruction());
  
	module->AddDeclaration(kernel);
	module->AddEntryFunction(kernel);
}

}
