%{
extern int yylineno;

#include "HorseIR/Tree/Program.h"
#include "Utils/Logger.h"

extern HorseIR::Program *program;

int yylex();
void yyerror(const char *s)
{
	Utils::Logger::LogError("(line " + std::to_string(yylineno) + ") " + s);
}
%}

%code requires {
	#include <string>
	#include <vector>

	#include "HorseIR/Tree/Import.h"
	#include "HorseIR/Tree/Method.h"
	#include "HorseIR/Tree/Module.h"
	#include "HorseIR/Tree/ModuleContent.h"
	#include "HorseIR/Tree/Parameter.h"
	#include "HorseIR/Tree/Program.h"
	#include "HorseIR/Tree/Expressions/CallExpression.h"
	#include "HorseIR/Tree/Expressions/CastExpression.h"
	#include "HorseIR/Tree/Expressions/Expression.h"
	#include "HorseIR/Tree/Expressions/Identifier.h"
	#include "HorseIR/Tree/Expressions/ModuleIdentifier.h"
	#include "HorseIR/Tree/Expressions/Literals/DateLiteral.h"
	#include "HorseIR/Tree/Expressions/Literals/IntLiteral.h"
	#include "HorseIR/Tree/Expressions/Literals/FloatLiteral.h"
	#include "HorseIR/Tree/Expressions/Literals/FunctionLiteral.h"
	#include "HorseIR/Tree/Expressions/Literals/StringLiteral.h"
	#include "HorseIR/Tree/Expressions/Literals/SymbolLiteral.h"
	#include "HorseIR/Tree/Statements/AssignStatement.h"
	#include "HorseIR/Tree/Statements/ReturnStatement.h"
	#include "HorseIR/Tree/Statements/Statement.h"
	#include "HorseIR/Tree/Types/Type.h"
	#include "HorseIR/Tree/Types/BasicType.h"
	#include "HorseIR/Tree/Types/DictionaryType.h"
	#include "HorseIR/Tree/Types/EnumerationType.h"
	#include "HorseIR/Tree/Types/FunctionType.h"
	#include "HorseIR/Tree/Types/KeyedTableType.h"
	#include "HorseIR/Tree/Types/ListType.h"
	#include "HorseIR/Tree/Types/TableType.h"
}

%locations
%error-verbose

%union {
	std::int64_t int_val;
	double float_val;
	std::string *string_val;

	std::vector<HorseIR::ModuleContent *> *module_contents;
	std::vector<HorseIR::Parameter *> *parameters;
	std::vector<HorseIR::Statement *> *statements;
	std::vector<HorseIR::Expression *> *expressions;
	std::vector<std::string> *string_list;
	std::vector<std::int64_t> *int_list;
	std::vector<double> *float_list;

	HorseIR::ModuleContent *module_content;
	HorseIR::Parameter *parameter;
	HorseIR::Type *type;
	HorseIR::BasicType *basic_type;
	HorseIR::Statement *statement;
	HorseIR::ModuleIdentifier *module_identifier;
	HorseIR::Expression *expression;
}

%token '(' ')' '{' '}' '<' '>'
%token tBOOL tI8 tI16 tI32 tI64 tF32 tF64 tCOMPLEX tSYMBOL tSTRING tMONTH tDATE tDATETIME tMINUTE tSECOND tTIME tFUNCTION
%token tLIST tDICTIONARY tENUM tTABLE tKTABLE
%token tMODULE tIMPORT tDEF tKERNEL tCHECKCAST tRETURN
%token <int_val> tINTVAL tDATEVAL
%token <float_val> tFLOATVAL
%token <string_val> tSTRINGVAL tSYMBOLVAL tIDENTIFIER

%type <module_contents> module_contents
%type <module_content> module_content
%type <parameters> parametersne parameters
%type <parameter> parameter
%type <type> type
%type <basic_type> int_type float_type
%type <statements> statements
%type <statement> statement
%type <module_identifier> module_identifier
%type <expressions> literals literalsne
%type <expression> expression call literal int_literal float_literal string_literal symbol_literal date_literal function_literal
%type <int_list> int_list date_list
%type <float_list> float_list
%type <string_list> string_list symbol_list

%start program

%%

program : tMODULE tIDENTIFIER '{' module_contents '}'                           { program = new HorseIR::Program({new HorseIR::Module(*$2, *$4)}); }
	;

module_contents : module_contents module_content                                { $1->push_back($2); $$ = $1; }
	        | %empty                                                        { $$ = new std::vector<HorseIR::ModuleContent *>(); }
                ;

module_content : tDEF tIDENTIFIER '(' parameters ')' ':' type '{' statements '}'{ $$ = new HorseIR::Method(*$2, *$4, $7, *$9); }
               | tKERNEL tIDENTIFIER '(' parameters ')' ':' type
                     '{' statements '}'                                         { $$ = new HorseIR::Method(*$2, *$4, $7, *$9, true); }
	       | tIMPORT module_identifier ';'                                  { $$ = new HorseIR::Import($2); }
               ;

parameters : parametersne                                                       { $$ = $1; }
	   | %empty                                                             { $$ = new std::vector<HorseIR::Parameter *>(); }
           ;

parametersne : parametersne ',' parameter                                       { $1->push_back($3); $$ = $1; }
	     | parameter                                                        { $$ = new std::vector<HorseIR::Parameter *>({$1}); }
             ;

parameter : tIDENTIFIER ':' type                                                { $$ = new HorseIR::Parameter(*$1, $3); }
	  ;

type : '?'                                                                      { $$ = new HorseIR::BasicType(HorseIR::BasicType::Kind::Wildcard); }
     | tBOOL                                                                    { $$ = new HorseIR::BasicType(HorseIR::BasicType::Kind::Bool); }
     | int_type                                                                 { $$ = $1; }
     | float_type                                                               { $$ = $1; }
     | tSYMBOL                                                                  { $$ = new HorseIR::BasicType(HorseIR::BasicType::Kind::Symbol); }
     | tSTRING                                                                  { $$ = new HorseIR::BasicType(HorseIR::BasicType::Kind::String); }
     | tCOMPLEX                                                                 { $$ = new HorseIR::BasicType(HorseIR::BasicType::Kind::Complex); }
     | tMONTH                                                                   { $$ = new HorseIR::BasicType(HorseIR::BasicType::Kind::Month); }
     | tDATE                                                                    { $$ = new HorseIR::BasicType(HorseIR::BasicType::Kind::Date); }
     | tDATETIME                                                                { $$ = new HorseIR::BasicType(HorseIR::BasicType::Kind::Datetime); }
     | tMINUTE                                                                  { $$ = new HorseIR::BasicType(HorseIR::BasicType::Kind::Minute); }
     | tSECOND                                                                  { $$ = new HorseIR::BasicType(HorseIR::BasicType::Kind::Second); }
     | tTIME                                                                    { $$ = new HorseIR::BasicType(HorseIR::BasicType::Kind::Time); }
     | tFUNCTION                                                                { $$ = new HorseIR::FunctionType(); }
     | tLIST '<' type '>'                                                       { $$ = new HorseIR::ListType($3); }
     | tTABLE                                                                   { $$ = new HorseIR::TableType(); }
     | tKTABLE                                                                  { $$ = new HorseIR::KeyedTableType(); }
     | tDICTIONARY '<' type ',' type '>'                                        { $$ = new HorseIR::DictionaryType($3, $5); }
     | tENUM '<' type ',' type '>'                                              { $$ = new HorseIR::EnumerationType($3, $5); }
     ;

int_type : tI8                                                                  { $$ = new HorseIR::BasicType(HorseIR::BasicType::Kind::Int8); }
         | tI16                                                                 { $$ = new HorseIR::BasicType(HorseIR::BasicType::Kind::Int16); }
         | tI32                                                                 { $$ = new HorseIR::BasicType(HorseIR::BasicType::Kind::Int32); }
         | tI64                                                                 { $$ = new HorseIR::BasicType(HorseIR::BasicType::Kind::Int64); }
	 ;

float_type : tF32                                                               { $$ = new HorseIR::BasicType(HorseIR::BasicType::Kind::Float32); }
           | tF64                                                               { $$ = new HorseIR::BasicType(HorseIR::BasicType::Kind::Float64); }
           ;

statements : statements statement                                               { $1->push_back($2); $$ = $1; }
	   | %empty                                                             { $$ = new std::vector<HorseIR::Statement *>(); }
           ;

statement : tIDENTIFIER ':' type '=' expression ';'                             { $$ = new HorseIR::AssignStatement(*$1, $3, $5); }
          | tRETURN tIDENTIFIER ';'                                             { $$ = new HorseIR::ReturnStatement(new HorseIR::Identifier(*$2)); }
          ;

expression : tCHECKCAST '(' expression ',' type ')'                             { $$ = new HorseIR::CastExpression($3, $5); }
	   | call                                                               { $$ = $1; }
           | literal                                                            { $$ = $1; }
           ; 

call : '@' module_identifier '(' literals ')'                                   { $$ = new HorseIR::CallExpression($2, *$4); }
     | '@' tIDENTIFIER '(' literals ')'                                         { $$ = new HorseIR::CallExpression(new HorseIR::ModuleIdentifier(*$2), *$4); }
     ;

module_identifier : tIDENTIFIER '.' tIDENTIFIER                                 { $$ = new HorseIR::ModuleIdentifier(*$1, *$3); }
                  | tIDENTIFIER '.' '*'                                         { $$ = new HorseIR::ModuleIdentifier(*$1, "*"); }
                  ;

literals : literalsne                                                           { $$ = $1; }
         | %empty                                                               { $$ = new std::vector<HorseIR::Expression *>(); }
         ;

literalsne : literalsne ',' literal                                             { $1->push_back($3); $$ = $1; }
           | literal                                                            { $$ = new std::vector<HorseIR::Expression *>({$1}); } 
           ;

literal : tIDENTIFIER                                                           { $$ = new HorseIR::Identifier(*$1); }
        | int_literal                                                           { $$ = $1; }
        | float_literal                                                         { $$ = $1; }
	| string_literal                                                        { $$ = $1; }
	| symbol_literal                                                        { $$ = $1; }
        | date_literal                                                          { $$ = $1; }
        | function_literal                                                      { $$ = $1; }
        ;

int_literal : tINTVAL ':' int_type                                              { $$ = HorseIR::CreateIntLiteral($1, $3); }
            | '(' int_list ')' ':' int_type                                     { $$ = HorseIR::CreateIntLiteral(*$2, $5); }

int_list : int_list ',' tINTVAL                                                 { $1->push_back($3); $$ = $1; } 
	 | tINTVAL                                                              { $$ = new std::vector<std::int64_t>({$1}); }
         ;

float_literal : tINTVAL ':' float_type                                          { $$ = HorseIR::CreateFloatLiteral($1, $3); }
              | '(' int_list ')' ':' float_type                                 { $$ = HorseIR::CreateFloatLiteral(*$2, $5); }
              | tFLOATVAL ':' float_type                                        { $$ = HorseIR::CreateFloatLiteral($1, $3); }
              | '(' float_list ')' ':' float_type                               { $$ = HorseIR::CreateFloatLiteral(*$2, $5); }
              ;

float_list : float_list ',' tFLOATVAL                                           { $1->push_back($3); $$ = $1; } 
           | tFLOATVAL                                                          { $$ = new std::vector<double>({$1}); }
           ;

string_literal : tSTRINGVAL ':' tSTRING                                         { $$ = new HorseIR::StringLiteral(*$1); }
               | '(' string_list ')' ':' tSTRING                                { $$ = new HorseIR::StringLiteral(*$2); }
               ;

string_list : string_list ',' tSTRINGVAL                                        { $1->push_back(*$3); $$ = $1; } 
	    | tSTRINGVAL                                                        { $$ = new std::vector<std::string>({*$1}); }
            ;

symbol_literal : tSYMBOLVAL ':' tSYMBOL                                         { $$ = new HorseIR::SymbolLiteral(*$1); }
               | '(' symbol_list ')' ':' tSYMBOL                                { $$ = new HorseIR::SymbolLiteral(*$2); }
               ;

symbol_list : symbol_list ',' tSYMBOLVAL                                        { $1->push_back(*$3); $$ = $1; } 
	    | tSYMBOLVAL                                                        { $$ = new std::vector<std::string>({*$1}); }
            ;

date_literal : tDATEVAL ':' tDATE                                               { $$ = new HorseIR::DateLiteral($1); }
             | '(' date_list ')' ':' tDATE                                      { $$ = new HorseIR::DateLiteral(*$2); }
             ;

date_list : date_list ',' tDATEVAL                                              { $1->push_back($3); $$ = $1; } 
	  | tDATEVAL                                                            { $$ = new std::vector<long>({$1}); }
          ;

function_literal : '@' module_identifier                                        { $$ = new HorseIR::FunctionLiteral($2); }
                 | '@' tIDENTIFIER                                              { $$ = new HorseIR::FunctionLiteral(new HorseIR::ModuleIdentifier(*$2)); }
                 ;

%%
