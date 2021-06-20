%{
extern int yylineno;

#include "HorseIR/Tree/Tree.h"
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

	#include "HorseIR/Tree/Tree.h"
}

%locations
%error-verbose

%union {
	std::int8_t  char_val;
	std::int64_t int_val;
	double       float_val;
	std::string  *string_val;

	HorseIR::ComplexValue  *complex_val;
	HorseIR::SymbolValue   *symbol_val;
	HorseIR::DatetimeValue *datetime_val;
	HorseIR::MonthValue    *month_val;
	HorseIR::DateValue     *date_val;
	HorseIR::MinuteValue   *minute_val;
	HorseIR::SecondValue   *second_val;
	HorseIR::TimeValue     *time_val;

	std::vector<HorseIR::Module *>        *modules;
	std::vector<HorseIR::ModuleContent *> *module_contents;
	std::vector<std::string>              *names;
	std::vector<HorseIR::Type *>          *types;
	std::vector<HorseIR::Parameter *>     *parameters;
	std::vector<HorseIR::Statement *>     *statements;
	std::vector<HorseIR::LValue *>        *lvalues;
	std::vector<HorseIR::Expression *>    *expressions;
	std::vector<HorseIR::Operand *>       *operands;

	std::vector<std::int8_t>              *char_list;
	std::vector<std::int64_t>             *int_list;
	std::vector<double>                   *float_list;
	std::vector<std::string>              *string_list;
	std::vector<HorseIR::ComplexValue *>  *complex_list;
	std::vector<HorseIR::SymbolValue *>   *symbol_list;
	std::vector<HorseIR::DatetimeValue *> *datetime_list;
	std::vector<HorseIR::MonthValue *>    *month_list;
	std::vector<HorseIR::DateValue *>     *date_list;
	std::vector<HorseIR::MinuteValue *>   *minute_list;
	std::vector<HorseIR::SecondValue *>   *second_list;
	std::vector<HorseIR::TimeValue *>     *time_list;

	HorseIR::Module              *module;
	HorseIR::ModuleContent       *module_content;
	HorseIR::Type                *type;
	HorseIR::ListType            *list_type;
	HorseIR::DictionaryType      *dictionary_type;
	HorseIR::EnumerationType     *enum_type;
	HorseIR::TableType           *table_type;
	HorseIR::KeyedTableType      *ktable_type;
	HorseIR::BasicType           *basic_type;
	HorseIR::Statement           *statement;
	HorseIR::BlockStatement      *block_statement;
	HorseIR::VariableDeclaration *declaration;
	HorseIR::Parameter           *parameter;
	HorseIR::Operand             *operand;
	HorseIR::FunctionLiteral     *function;
	HorseIR::LValue              *lvalue;
	HorseIR::Expression          *expression;
	HorseIR::Identifier          *identifier;
}

%token tARROW

%token tBOOL tCHAR tI8 tI16 tI32 tI64 tF32 tF64 tCOMPLEX tSYMBOL tSTRING tMONTH tDATE tDATETIME tMINUTE tSECOND tTIME tFUNCTION
%token tLIST tDICTIONARY tENUM tTABLE tKTABLE
%token tMODULE tIMPORT tGLOBAL tDEF tKERNEL tCHECKCAST tIF tELSE tWHILE tREPEAT tVAR tRETURN tBREAK tCONTINUE

%token <char_val> tCHARVAL
%token <int_val> tINTVAL
%token <float_val> tFLOATVAL
%token <complex_val> tCOMPLEXVAL
%token <string_val> tSTRINGVAL tIDENTIFIER
%token <symbol_val> tSYMBOLVAL
%token <datetime_val> tDATETIMEVAL
%token <month_val> tMONTHVAL
%token <date_val> tDATEVAL
%token <minute_val> tMINUTEVAL
%token <second_val> tSECONDVAL
%token <time_val> tTIMEVAL

%type <modules> modules
%type <module> module
%type <module_contents> module_contents
%type <module_content> module_content global_declaration function import_directive
%type <names> names
%type <declaration> variable_declaration
%type <parameters> parametersne parameters
%type <parameter> parameter
%type <types> return_types types
%type <type> type
%type <basic_type> int_type float_type
%type <list_type> list_type
%type <dictionary_type> dictionary_type
%type <enum_type> enum_type
%type <table_type> table_type
%type <ktable_type> ktable_type
%type <statements> statements declaration_statement
%type <statement> statement
%type <block_statement> control_block
%type <lvalues> lvalues
%type <lvalue> lvalue
%type <operands> operands operandsne
%type <identifier> identifier
%type <expression> expression
%type <operand> operand bool_literal char_literal int_literal float_literal complex_literal string_literal symbol_literal datetime_literal date_literal month_literal minute_literal second_literal time_literal
%type <function> function_literal

%type <char_list> char_list
%type <int_list> int_list
%type <float_list> float_list
%type <complex_list> complex_list
%type <string_list> string_list
%type <symbol_list> symbol_list
%type <datetime_list> datetime_list
%type <month_list> month_list
%type <date_list> date_list
%type <minute_list> minute_list
%type <second_list> second_list
%type <time_list> time_list

%nonassoc ')'
%nonassoc tELSE

%start program

%%

program : modules                                                               { program = new HorseIR::Program(*$1); }
	;

modules : modules module                                                        { $1->push_back($2); $$ = $1; }
	| %empty                                                                { $$ = new std::vector<HorseIR::Module *>(); }
        ;

module : tMODULE tIDENTIFIER '{' module_contents '}'                            { $$ = new HorseIR::Module(*$2, *$4); }
       ;

module_contents : module_contents module_content                                { $1->push_back($2); $$ = $1; }
	        | %empty                                                        { $$ = new std::vector<HorseIR::ModuleContent *>(); }
                ;

module_content : global_declaration                                             { $$ = $1; }
	       | function                                                       { $$ = $1; }
               | import_directive ';'                                           { $$ = $1; }
               ;

global_declaration : tGLOBAL variable_declaration '=' operand ';'               { $$ = new HorseIR::GlobalDeclaration($2, $4); }
                   ;

variable_declaration : tIDENTIFIER ':' type                                     { $$ = new HorseIR::VariableDeclaration(*$1, $3); }
		     ;

function : tDEF tIDENTIFIER '(' parameters ')' return_types '{' statements '}'     { $$ = new HorseIR::Function(*$2, *$4, *$6, *$8); }
         | tKERNEL tIDENTIFIER '(' parameters ')' return_types '{' statements '}'  { $$ = new HorseIR::Function(*$2, *$4, *$6, *$8, true); }
         ;

return_types : ':' types                                                        { $$ = $2; }
	     | %empty                                                           { $$ = new std::vector<HorseIR::Type *>(); }
             ;

types : types ',' type                                                          { $1->push_back($3); $$ = $1; }
      | type                                                                    { $$ = new std::vector<HorseIR::Type *>({$1}); }
      ;

import_directive : tIMPORT tIDENTIFIER '.' '*'                                  { $$ = new HorseIR::ImportDirective(*$2, "*"); }
		 | tIMPORT tIDENTIFIER '.' tIDENTIFIER                          { $$ = new HorseIR::ImportDirective(*$2, *$4); }
                 | tIMPORT tIDENTIFIER '.' '{' names '}'                        { $$ = new HorseIR::ImportDirective(*$2, *$5); }
                 ;

names : names ',' tIDENTIFIER                                                   { $1->push_back(*$3); $$ = $1; }
      | tIDENTIFIER                                                             { $$ = new std::vector<std::string>({*$1}); }
      ;

parameters : parametersne                                                       { $$ = $1; }
	   | %empty                                                             { $$ = new std::vector<HorseIR::Parameter *>(); }
           ;

parametersne : parametersne ',' parameter                                       { $1->push_back($3); $$ = $1; }
	     | parameter                                                        { $$ = new std::vector<HorseIR::Parameter *>({$1}); }
             ;

parameter : tIDENTIFIER ':' type                                                { $$ = new HorseIR::Parameter(*$1, $3); }
	  ;

type : '?'                                                                      { $$ = new HorseIR::WildcardType(); }
     | tBOOL                                                                    { $$ = new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Boolean); }
     | tCHAR                                                                    { $$ = new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Char); }
     | int_type                                                                 { $$ = $1; }
     | float_type                                                               { $$ = $1; }
     | tSYMBOL                                                                  { $$ = new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Symbol); }
     | tSTRING                                                                  { $$ = new HorseIR::BasicType(HorseIR::BasicType::BasicKind::String); }
     | tCOMPLEX                                                                 { $$ = new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Complex); }
     | tMONTH                                                                   { $$ = new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Month); }
     | tDATE                                                                    { $$ = new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Date); }
     | tDATETIME                                                                { $$ = new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Datetime); }
     | tMINUTE                                                                  { $$ = new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Minute); }
     | tSECOND                                                                  { $$ = new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Second); }
     | tTIME                                                                    { $$ = new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Time); }
     | tFUNCTION                                                                { $$ = new HorseIR::FunctionType(); }
     | list_type                                                                { $$ = $1; }
     | table_type                                                               { $$ = $1; }
     | ktable_type                                                              { $$ = $1; }
     | dictionary_type                                                          { $$ = $1; }
     | enum_type                                                                { $$ = $1; }
     ;

int_type : tI8                                                                  { $$ = new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int8); }
         | tI16                                                                 { $$ = new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int16); }
         | tI32                                                                 { $$ = new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int32); }
         | tI64                                                                 { $$ = new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64); }
	 ;

float_type : tF32                                                               { $$ = new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Float32); }
           | tF64                                                               { $$ = new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Float64); }
           ;

list_type : tLIST '<' types '>'                                                 { $$ = new HorseIR::ListType(*$3); }
	  ;

dictionary_type : tDICTIONARY '<' type ',' type '>'                             { $$ = new HorseIR::DictionaryType($3, $5); }
		;

enum_type : tENUM '<' type '>'                                                  { $$ = new HorseIR::EnumerationType($3); }
	  ;

table_type : tTABLE                                                             { $$ = new HorseIR::TableType(); }
	   ;

ktable_type : tKTABLE                                                           { $$ = new HorseIR::KeyedTableType(); }
	    ;

statements : statements statement                                               { $1->push_back($2); $$ = $1; }
           | statements declaration_statement                                   { $1->insert($1->end(), $2->begin(), $2->end()); $$ = $1; }
	   | %empty                                                             { $$ = new std::vector<HorseIR::Statement *>(); }
           ;

declaration_statement : tVAR names ':' type ';'                                 { $$ = HorseIR::CreateDeclarationStatements(*$2, $4); }
		      ;

statement : lvalues '=' expression ';'                                          { $$ = new HorseIR::AssignStatement(*$1, $3, @2.first_line); }
          | expression ';'                                                      { $$ = new HorseIR::ExpressionStatement($1, @2.first_line); }
          | tIF '(' operand ')' control_block                                   { $$ = new HorseIR::IfStatement($3, $5, nullptr, @1.first_line); }
          | tIF '(' operand ')' control_block tELSE control_block               { $$ = new HorseIR::IfStatement($3, $5, $7, @1.first_line); }
          | tWHILE '(' operand ')' control_block                                { $$ = new HorseIR::WhileStatement($3, $5, @1.first_line); }
          | tREPEAT '(' operand ')' control_block                               { $$ = new HorseIR::RepeatStatement($3, $5, @1.first_line); }
          | tRETURN operandsne ';'                                              { $$ = new HorseIR::ReturnStatement(*$2, @1.first_line); }
          | tBREAK ';'                                                          { $$ = new HorseIR::BreakStatement(@1.first_line); }
          | tCONTINUE ';'                                                       { $$ = new HorseIR::ContinueStatement(@1.first_line); }
          ;

control_block : '{' statements '}'                                              { $$ = new HorseIR::BlockStatement(*$2); }
	      | statement                                                       { $$ = new HorseIR::BlockStatement({$1}); }
              ;

lvalues : lvalues ',' lvalue                                                    { $1->push_back($3); $$ = $1; }
	| lvalue                                                                { $$ = new std::vector<HorseIR::LValue *>({$1}); }
        ;

lvalue : identifier                                                             { $$ = $1; }
       | variable_declaration                                                   { $$ = $1; }
       ;

expression : tCHECKCAST '(' expression ',' type ')'                             { $$ = new HorseIR::CastExpression($3, $5); }
           | function_literal '(' operands ')'                                  { $$ = new HorseIR::CallExpression($1, *$3); }
           | operand                                                            { $$ = $1; }
           ; 

operands : operandsne                                                           { $$ = $1; }
         | %empty                                                               { $$ = new std::vector<HorseIR::Operand *>(); }
         ;

operandsne : operandsne ',' operand                                             { $1->push_back($3); $$ = $1; }
           | operand                                                            { $$ = new std::vector<HorseIR::Operand *>({$1}); } 
           ;

operand : identifier                                                            { $$ = $1; }
	| bool_literal                                                          { $$ = $1; }
        | char_literal                                                          { $$ = $1; }
        | int_literal                                                           { $$ = $1; }
        | float_literal                                                         { $$ = $1; }
        | complex_literal                                                       { $$ = $1; }
	| string_literal                                                        { $$ = $1; }
	| symbol_literal                                                        { $$ = $1; }
        | datetime_literal                                                      { $$ = $1; }
        | month_literal                                                         { $$ = $1; }
        | date_literal                                                          { $$ = $1; }
        | minute_literal                                                        { $$ = $1; }
        | second_literal                                                        { $$ = $1; }
        | time_literal                                                          { $$ = $1; }
	| function_literal                                                      { $$ = $1; }
	| function_literal ':' tFUNCTION                                        { $$ = $1; }
        ;

function_literal : '@' identifier                                               { $$ = new HorseIR::FunctionLiteral($2); }
                 ;

identifier : tIDENTIFIER '.' tIDENTIFIER                                        { $$ = new HorseIR::Identifier(*$1, *$3); }
           | tIDENTIFIER                                                        { $$ = new HorseIR::Identifier(*$1); }
           ;

bool_literal : tINTVAL ':' tBOOL                                                { $$ = HorseIR::CreateBooleanLiteral($1); }
	     | '(' int_list ')' ':' tBOOL                                       { $$ = HorseIR::CreateBooleanLiteral(*$2); }
             ;

char_literal : tCHARVAL ':' tCHAR                                               { $$ = new HorseIR::CharLiteral($1); }
	     | '(' char_list ')' ':' tCHAR                                      { $$ = new HorseIR::CharLiteral(*$2); }
             ;

char_list : char_list ',' tCHARVAL                                              { $1->push_back($3); $$ = $1; }
	  | tCHARVAL                                                            { $$ = new std::vector<std::int8_t>({$1}); }
          ;

int_literal : tINTVAL ':' int_type                                              { $$ = HorseIR::CreateIntLiteral($1, $3); }
            | '(' int_list ')' ':' int_type                                     { $$ = HorseIR::CreateIntLiteral(*$2, $5); }

int_list : int_list ',' tINTVAL                                                 { $1->push_back($3); $$ = $1; } 
	 | tINTVAL                                                              { $$ = new std::vector<std::int64_t>({$1}); }
         ;

float_literal : tFLOATVAL ':' float_type                                        { $$ = HorseIR::CreateFloatLiteral($1, $3); }
              | tINTVAL ':' float_type                                          { $$ = HorseIR::CreateFloatLiteral($1, $3); }
              | '(' float_list ')' ':' float_type                               { $$ = HorseIR::CreateFloatLiteral(*$2, $5); }
              ;

float_list : float_list ',' tFLOATVAL                                           { $1->push_back($3); $$ = $1; } 
           | tFLOATVAL                                                          { $$ = new std::vector<double>({$1}); }
           ;

complex_literal : tCOMPLEXVAL ':' tCOMPLEX                                      { $$ = new HorseIR::ComplexLiteral($1); }
                | '(' complex_list ')' ':' tCOMPLEX                             { $$ = new HorseIR::ComplexLiteral(*$2); }

complex_list : complex_list ',' tCOMPLEXVAL                                     { $1->push_back($3); $$ = $1; } 
             | tCOMPLEXVAL                                                      { $$ = new std::vector<HorseIR::ComplexValue *>({$1}); }
             ;

string_literal : tSTRINGVAL ':' tSTRING                                         { $$ = new HorseIR::StringLiteral(*$1); }
               | '(' string_list ')' ':' tSTRING                                { $$ = new HorseIR::StringLiteral(*$2); }
               ;

string_list : string_list ',' tSTRINGVAL                                        { $1->push_back(*$3); $$ = $1; } 
	    | tSTRINGVAL                                                        { $$ = new std::vector<std::string>({*$1}); }
            ;

symbol_literal : tSYMBOLVAL ':' tSYMBOL                                         { $$ = new HorseIR::SymbolLiteral($1); }
               | '(' symbol_list ')' ':' tSYMBOL                                { $$ = new HorseIR::SymbolLiteral(*$2); }
               ;

symbol_list : symbol_list ',' tSYMBOLVAL                                        { $1->push_back($3); $$ = $1; } 
	    | tSYMBOLVAL                                                        { $$ = new std::vector<HorseIR::SymbolValue *>({$1}); }
            ;

datetime_literal : tDATETIMEVAL ':' tDATETIME                                   { $$ = new HorseIR::DatetimeLiteral($1); }
		 | '(' datetime_list ')' ':' tDATETIME                          { $$ = new HorseIR::DatetimeLiteral(*$2); }
                 ;

datetime_list : datetime_list ',' tDATETIMEVAL                                  { $1->push_back($3); $$ = $1; }
	      | tDATETIMEVAL                                                    { $$ = new std::vector<HorseIR::DatetimeValue *>({$1}); }
              ;

month_literal : tMONTHVAL ':' tMONTH                                            { $$ = new HorseIR::MonthLiteral($1); }
              | '(' month_list ')' ':' tMONTH                                   { $$ = new HorseIR::MonthLiteral(*$2); }
              ;

month_list : month_list ',' tMONTHVAL                                           { $1->push_back($3); $$ = $1; } 
	   | tMONTHVAL                                                          { $$ = new std::vector<HorseIR::MonthValue *>({$1}); }
           ;

date_literal : tDATEVAL ':' tDATE                                               { $$ = new HorseIR::DateLiteral($1); }
             | '(' date_list ')' ':' tDATE                                      { $$ = new HorseIR::DateLiteral(*$2); }
             ;

date_list : date_list ',' tDATEVAL                                              { $1->push_back($3); $$ = $1; } 
	  | tDATEVAL                                                            { $$ = new std::vector<HorseIR::DateValue *>({$1}); }
          ;

minute_literal : tMINUTEVAL ':' tMINUTE                                         { $$ = new HorseIR::MinuteLiteral($1); }
	       | '(' minute_list ')' ':' tMINUTE                                { $$ = new HorseIR::MinuteLiteral(*$2); }
               ;

minute_list : minute_list ',' tMINUTEVAL                                        { $1->push_back($3); $$ = $1; }
	    | tMINUTEVAL                                                        { $$ = new std::vector<HorseIR::MinuteValue *>({$1}); }
            ;

second_literal : tSECONDVAL ':' tSECOND                                         { $$ = new HorseIR::SecondLiteral($1); }
	       | '(' second_list ')' ':' tSECOND                                { $$ = new HorseIR::SecondLiteral(*$2); }
               ;

second_list : second_list ',' tSECONDVAL                                        { $1->push_back($3); $$ = $1; }
	    | tSECONDVAL                                                        { $$ = new std::vector<HorseIR::SecondValue *>({$1}); }
            ;

time_literal : tTIMEVAL ':' tTIME                                               { $$ = new HorseIR::TimeLiteral($1); }
	     | '(' time_list ')' ':' tTIME                                      { $$ = new HorseIR::TimeLiteral(*$2); }
             ;

time_list : time_list ',' tTIMEVAL                                              { $1->push_back($3); $$ = $1; }
	  | tTIMEVAL                                                            { $$ = new std::vector<HorseIR::TimeValue *>({$1}); }
          ;

%%
