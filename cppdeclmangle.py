import argparse, re


class Keys:
    VFTABLE = 'vftable'

    L_ANG_BRACKET = '<'
    R_ANG_BRACKET = '>'

    L_SQR_BRACKET = '['
    R_SQR_BRACKET = ']'

    L_PAREN = '('
    R_PAREN = ')'

    SEPARATOR = ','

    SCOPE_RESOLUTION = '::'

    PUBLIC = 'public'
    PRIVATE = 'private'
    PROTECTED = 'protected'
    ACCESS_SCOPE = ':'

    CONST = 'const'
    VOLATILE = 'volatile'

    STATIC = 'static'
    VIRTUAL = 'virtual'
    CDECL = '__cdecl'
    STDCALL = '__stdcall'
    FASTCALL = '__fastcall'
    
    VOID = 'void'
    BOOL = 'bool'
    FLOAT = 'float'
    DOUBLE = 'double'
    SIGNED = 'signed'
    UNSIGNED = 'unsigned'
    CHAR = 'char'
    SHORT = 'short'
    INT = 'int'
    LONG = 'long'
    INT64 = '__int64'
    WCHAR = 'wchar_t'
    CLASS = 'class'
    STRUCT = 'struct'
    UNION = 'union'

    PTR = '*'
    REF = '&'
    RVAL_REF = '&&'
    PTR32 = '__ptr32'
    PTR64 = '__ptr64'
 
    DESTRUCTOR = '~'
    OPERATOR = 'operator'
    NEW = 'new'
    ASSIGN = '='
    RSHIFT = '>>'
    LSHIFT = '<<'
    NOT = '!'
    EQUALS = '=='
    NOT_EQUALS = '!='
    ARROW = '->'
    INCREMENT = '++'
    DECREMENT = '--'
    SUBTRACT = '-'
    ADD = '+'
    PTR_TO_MEMBER = '->*'
    DIVIDE = '/'
    MODULUS = '%'
    LESS_THAN = '<'
    LESS_OR_EQUALS = '<='
    GREATER_THAN = '>'
    GREATER_OR_EQUALS = '>='
    COMMA = ','
    MULT_ASSIGN = '*='
    ADD_ASSIGN = '+='
    SUBTRACT_ASSIGN = '-='
    DIVIDE_ASSIGN = '/='


class lazyattr: # lazy resolution class attribute
    def __init__(self, getter):
        self.getter = getter
        
    def __get__(self, instance, owner):
        value = self.getter(owner)
        setattr(owner, self.getter.__name__, value)
        return value


class MetaNode(type):
    def __call__(cls, string):
        if hasattr(cls, 'producers'): 
            for producer in cls.producers:
                if product := producer(string):
                    return product
            if 'build' not in cls.__dict__:
                return None
        
        match = cls.regex.fullmatch(string)
        if not match:
            return None
        instance = cls.__new__(cls)
        if 'build' in cls.__dict__:
            if instance.build(match) == Node.BUILD_ERROR:
                return None
        return instance
    
    def __instancecheck__(cls, instance):
        if type.__instancecheck__(cls, instance):
            return True
        
        if hasattr(cls, '_flattened_producers'):
            return any(type.__instancecheck__(producer, instance) for producer in cls._flattened_producers)
        return False


class Node(metaclass=MetaNode):
    BUILD_ERROR = object()
 
    @lazyattr
    def genericPattern(cls):
        return re.sub(r'\(\?P<\w+>', '(?:', cls.regex.pattern)
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        
        if hasattr(cls, 'producers') and cls.producers:
            def flatten_producers(producers):
                terminals = []
                for producer in producers:
                    if hasattr(producer, 'producers') and producer.producers:
                        if 'build' in producer.__dict__:
                            terminals.append(producer)
                        terminals.extend(flatten_producers(producer.producers))
                    else:
                        terminals.append(producer)
                return tuple(set(terminals))
            cls._flattened_producers = flatten_producers(cls.producers)
    
    def __eq__(self, other):
        return self is other or str(self) == str(other)
    
    def __hash__(self):
        return hash(str(self))


class AccessSpecifier(Node):
    @lazyattr
    def regex(cls):
        return re.compile(rf'{Keys.PUBLIC}|{Keys.PRIVATE}|{Keys.PROTECTED}', re.VERBOSE)
    
    @lazyattr
    def PUBLIC(cls):
        return cls(Keys.PUBLIC)
    
    @lazyattr
    def PRIVATE(cls):
        return cls(Keys.PRIVATE)
    
    @lazyattr
    def PROTECTED(cls):
        return cls(Keys.PROTECTED)
    
    def build(self, match):
        self.specifier = match.group()

    def __str__(self):
        return self.specifier


class ResolutionSpecifier(Node):
    @lazyattr
    def regex(cls):
        return re.compile(rf'{Keys.STATIC}|{Keys.VIRTUAL}', re.VERBOSE)
    
    @lazyattr
    def STATIC(cls):
        return cls(Keys.STATIC)
    
    @lazyattr
    def VIRTUAL(cls):
        return cls(Keys.VIRTUAL)
    
    def build(self, match):
        self.specifier = match.group()
    
    def __str__(self):
        return self.specifier


class CallConvention(Node):
    @lazyattr
    def regex(cls):
        return re.compile(rf'{Keys.CDECL}|{Keys.STDCALL}|{Keys.FASTCALL}', re.VERBOSE)

    @lazyattr
    def CDECL(cls):
        return cls(Keys.CDECL)
    
    @lazyattr
    def STDCALL(cls):
        return cls(Keys.STDCALL)

    def build(self, match):
        self.specifier = match.group()
    
    def __str__(self):
        return self.specifier


class CVQualifiers(Node):
    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            (?P<first>{Keys.CONST}|{Keys.VOLATILE})
            (?:\s+(?P<last>{Keys.CONST}|{Keys.VOLATILE}))?
        ''', re.VERBOSE)
    
    @lazyattr
    def CONST(cls):
        return cls(Keys.CONST)
    
    @lazyattr
    def VOLATILE(cls):
        return cls(Keys.VOLATILE)
    
    @lazyattr
    def CONST_VOLATILE(cls):
        return cls(f'{Keys.CONST} {Keys.VOLATILE}')
    
    def build(self, match):
        if len(set(match.groups())) < len(match.groups()):
            return self.BUILD_ERROR
        
        self.qualifiers = [cvQual for cvQual in match.groups() if cvQual]

    def __iter__(self):
        for i in range(len(self.qualifiers)):
            yield self.qualifiers[i]
    
    def __getitem__(self, key):
        return self.qualifiers[key]
    
    def __add__(self, other):
        return CVQualifiers(f'{self} {other}'.strip())
    
    def __str__(self):
        return ' '.join(self.qualifiers).strip()


class ConstantExpression(Node):
    @lazyattr
    def regex(cls):
        return re.compile(r'\d+', re.VERBOSE)
    
    def build(self, match):
        self.value = match.group()
    
    def __str__(self):
        return self.value


class OverloadableOperator(Node):
    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            {Keys.NEW}|\{Keys.ASSIGN}|{Keys.RSHIFT}|{Keys.LSHIFT}|{Keys.NOT}|{Keys.EQUALS}|{Keys.NOT_EQUALS}|\{Keys.ARROW}|
            {re.escape(Keys.PTR)}|\{Keys.INCREMENT}|\{Keys.DECREMENT}|\{Keys.SUBTRACT}|\{Keys.ADD}|\{Keys.REF}|{re.escape(Keys.PTR_TO_MEMBER)}|
            \{Keys.DIVIDE}|\{Keys.MODULUS}|\{Keys.LESS_THAN}|\{Keys.LESS_OR_EQUALS}|\{Keys.GREATER_THAN}|\{Keys.GREATER_OR_EQUALS}|
            \{Keys.COMMA}|\{Keys.MULT_ASSIGN}|\{Keys.ADD_ASSIGN}|\{Keys.SUBTRACT_ASSIGN}|\{Keys.DIVIDE_ASSIGN}
        ''', re.VERBOSE)
    
    @lazyattr
    def NEW(cls):
        return cls(Keys.NEW)
    
    @lazyattr
    def ASSIGN(cls):
        return cls(Keys.ASSIGN)
    
    @lazyattr
    def LSHIFT(cls):
        return cls(Keys.LSHIFT)
    
    @lazyattr
    def RSHIFT(cls):
        return cls(Keys.RSHIFT)
    
    @lazyattr
    def NOT(cls):
        return cls(Keys.NOT)
    
    @lazyattr
    def EQUALS(cls):
        return cls(Keys.EQUALS)
    
    @lazyattr
    def NOT_EQUALS(cls):
        return cls(Keys.NOT_EQUALS)
    
    @lazyattr
    def ARROW(cls):
        return cls(Keys.ARROW)
    
    @lazyattr
    def PTR(cls):
        return cls(Keys.PTR)
    
    @lazyattr
    def INCREMENT(cls):
        return cls(Keys.INCREMENT)
    
    @lazyattr
    def DECREMENT(cls):
        return cls(Keys.DECREMENT)
    
    @lazyattr
    def SUBTRACT(cls):
        return cls(Keys.SUBTRACT)
    
    @lazyattr
    def ADD(cls):
        return cls(Keys.ADD)
    
    @lazyattr
    def PTR_TO_MEMBER(cls):
        return cls(Keys.PTR_TO_MEMBER)
    
    @lazyattr
    def DIVIDE(cls):
        return cls(Keys.DIVIDE)
    
    @lazyattr
    def MODULUS(cls):
        return cls(Keys.MODULUS)

    @lazyattr
    def LESS_THAN(cls):
        return cls(Keys.LESS_THAN)
    
    @lazyattr
    def LESS_OR_EQUALS(cls):
        return cls(Keys.LESS_OR_EQUALS)
    
    @lazyattr
    def GREATER_THAN(cls):
        return cls(Keys.GREATER_THAN)
    
    @lazyattr
    def GREATER_OR_EQUALS(cls):
        return cls(Keys.GREATER_OR_EQUALS)

    @lazyattr
    def COMMA(cls):
        return cls(Keys.COMMA)
    
    @lazyattr
    def MULT_ASSIGN(cls):
        return cls(Keys.MULT_ASSIGN)
    
    @lazyattr
    def ADD_ASSIGN(cls):
        return cls(Keys.ADD_ASSIGN)
    
    @lazyattr
    def SUBTRACT_ASSIGN(cls):
        return cls(Keys.SUBTRACT_ASSIGN)
    
    @lazyattr
    def DIVIDE_ASSIGN(cls):
        return cls(Keys.DIVIDE_ASSIGN)
    
    def build(self, match):
        self.operator = match.group()

    def __str__(self):
        return self.operator


class Identifier(Node):
    @lazyattr
    def regex(cls):
        return re.compile(rf'[_a-zA-Z]\w*', re.VERBOSE)

    def build(self, match):
        self.name = match.group()

    def __str__(self):
        return self.name


class OperatorFunctionID(Node):
    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            (?P<scope>{NestedNameSpecifier.genericPattern})?
            \b{Keys.OPERATOR}\b\s*
            (?P<overloadableOp>{OverloadableOperator.genericPattern})
        ''', re.VERBOSE)

    def build(self, match):
        self.scope = (
            NestedNameSpecifier(scope)
            if (scope := match.group('scope'))
            else None
        )
        self.identifier = OverloadableOperator(match.group('overloadableOp'))

    def __str__(self):
        return f"{self.scope or ''}{Keys.OPERATOR} {self.identifier}"


class TemplateArgsList(Node):
    @lazyattr
    def regex(cls):
        return re.compile(rf'{Keys.L_ANG_BRACKET}.*{Keys.R_ANG_BRACKET}', re.VERBOSE) # this be a problem

    @staticmethod
    def split_arguments(args_list: str):
        if not args_list:
            return []
        result = []
        buffer = ""
        depth = 0
        
        for char in args_list[1:-1]:
            if char in (Keys.L_ANG_BRACKET, Keys.L_PAREN):
                depth += 1
            elif char in (Keys.R_ANG_BRACKET, Keys.R_PAREN):
                depth = max(depth - 1, 0)
            elif char == Keys.SEPARATOR and depth == 0:
                result.append(buffer.strip())
                buffer = ""
                continue

            buffer += char

        if buffer:
            result.append(buffer.strip())

        return result

    def build(self, match):
        args = self.split_arguments(match.group())
        self.args_list = [TemplateArgument(arg) for arg in args]

    def __len__(self):
        return len(self.args_list)

    def __getitem__(self, key):
        return self.args_list[key]

    def __str__(self):
        return (
            f'{Keys.L_ANG_BRACKET}'
            f'{f'{Keys.SEPARATOR}'.join(str(arg) for arg in self.args_list)}'
            f'{Keys.R_ANG_BRACKET}'
        )


class SimpleTemplateID(Node):
    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            (?P<identifier>{Identifier.genericPattern})
            (?P<argsList>{TemplateArgsList.genericPattern})
        ''', re.VERBOSE)

    def build(self, match):
        self.identifier = Identifier(match.group('identifier'))
        self.template_args_list = TemplateArgsList(match.group('argsList'))

    def __str__(self):
        return f"{self.identifier}{self.template_args_list}"


class OperatorFunctionTemplateID(Node):
    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            (?P<operatorFuncID>{OperatorFunctionID.genericPattern})
            (?P<argsList>{TemplateArgsList.genericPattern})
        ''', re.VERBOSE)

    def build(self, match):
        self.operator = OverloadableOperator(match.group('operatorFuncID'))
        self.template_args_list = TemplateArgsList(match.group('argsList'))

    def __str__(self):
        return f"operator{self.operator}{self.template_args_list}"


class TemplateID(Node):
    producers = (SimpleTemplateID, OperatorFunctionTemplateID)

    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            (?:
                {Identifier.genericPattern}
                |
                \b{Keys.OPERATOR}\b\s*{OverloadableOperator.genericPattern}
            )
            {TemplateArgsList.genericPattern}
        ''', re.VERBOSE)


class UnqualifiedID(Node):
    producers = (TemplateID, OperatorFunctionID, Identifier)

    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            (?:
                {Identifier.genericPattern}
                |
                \b{Keys.OPERATOR}\b\s*{OverloadableOperator.genericPattern}
            )
            (?:{TemplateArgsList.genericPattern})?
        ''', re.VERBOSE)


class NestedNameSpecifier(Node):
    @lazyattr
    def regex(cls):
        return re.compile(rf'(?:{UnqualifiedID.genericPattern}{Keys.SCOPE_RESOLUTION})+', re.VERBOSE)
    
    @staticmethod
    def rpartition_scope(scope):
        if not scope:
            return '', '', scope
        depth = 0

        for i in range(len(scope) - 1, 0, -1):
            char = scope[i]

            if char == Keys.R_ANG_BRACKET:
                depth += 1
            elif char == Keys.L_ANG_BRACKET:
                depth = max(depth - 1, 0)

            if depth == 0 and scope[i-1:i+1] == Keys.SCOPE_RESOLUTION:
                return scope[:i-1], Keys.SCOPE_RESOLUTION, scope[i+1:]

        return "", "", scope
    
    def build(self, match):
        previous, __, name = self.rpartition_scope(match.group().rstrip(Keys.SCOPE_RESOLUTION))
        self.identifier = UnqualifiedID(name)
        self.scope = NestedNameSpecifier(f"{previous}{Keys.SCOPE_RESOLUTION}") if previous else None

    def __str__(self):
        return f'{self.scope or ''}{self.identifier}{Keys.SCOPE_RESOLUTION}'


class FundamentalTypeSpecifier(Node):
    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            {Keys.VOID}|{Keys.BOOL}|{Keys.FLOAT}|{Keys.DOUBLE}
            |
            (?:(?P<signage>{Keys.SIGNED}|{Keys.UNSIGNED})\s+)?
            (?:{Keys.CHAR}|{Keys.WCHAR}|{Keys.SHORT}|{Keys.INT}|{Keys.LONG}|{Keys.INT64})
        ''', re.VERBOSE)

    @lazyattr
    def VOID(cls):
        return cls(Keys.VOID)

    @lazyattr
    def BOOL(cls):
        return cls(Keys.BOOL)

    @lazyattr
    def FLOAT(cls):
        return cls(Keys.FLOAT)

    @lazyattr
    def DOUBLE(cls):
        return cls(Keys.DOUBLE)
    
    @lazyattr
    def SCHAR(cls):
        return cls(f'{Keys.SIGNED} {Keys.CHAR}')
    
    @lazyattr
    def CHAR(cls):
        return cls(Keys.CHAR)
    
    @lazyattr
    def UCHAR(cls):
        return cls(f'{Keys.UNSIGNED} {Keys.CHAR}')
    
    @lazyattr
    def WCHAR(cls):
        return cls(Keys.WCHAR)
    
    @lazyattr
    def SHORT(cls):
        return cls(Keys.SHORT)
    
    @lazyattr
    def USHORT(cls):
        return cls(f'{Keys.UNSIGNED} {Keys.SHORT}')

    @lazyattr
    def INT(cls):
        return cls(Keys.INT)
    
    @lazyattr
    def UINT(cls):
        return cls(f'{Keys.UNSIGNED} {Keys.INT}')
    
    @lazyattr
    def LONG(cls):
        return cls(Keys.LONG)
    
    @lazyattr
    def ULONG(cls):
        return cls(f'{Keys.UNSIGNED} {Keys.LONG}')
    
    @lazyattr
    def INT64(cls):
        return cls(Keys.INT64)
    
    @lazyattr
    def UINT64(cls):
        return cls(f'{Keys.UNSIGNED} {Keys.INT64}')
    
    def build(self, match):
        self.specifier = match.group()
        self.signage = match.group('signage')
        
        if Keys.CHAR not in self.specifier and self.signage == Keys.SIGNED:
            self.specifier = self.specifier.replace(Keys.SIGNED, '').strip()
    
    def __str__(self):
        return self.specifier


class ClassKey(Node):
    @lazyattr
    def regex(cls):
        return re.compile(rf'{Keys.CLASS}|{Keys.STRUCT}|{Keys.UNION}', re.VERBOSE)
    
    @lazyattr
    def CLASS(cls):
        return cls(Keys.CLASS)
    
    @lazyattr
    def STRUCT(cls):
        return cls(Keys.STRUCT)
    
    @lazyattr
    def UNION(cls):
        return cls(Keys.UNION)
    
    def build(self, match):
        self.key: str = match.group()
    
    def __str__(self):
        return self.key


class ElaboratedTypeSpecifier(Node):
    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            (?P<classKey>{ClassKey.genericPattern})\s+
            (?P<name>(?:{NestedNameSpecifier.genericPattern})?{UnqualifiedID.genericPattern})
        ''', re.VERBOSE)

    def build(self, match):
        self.class_key = ClassKey(match.group('classKey'))
        self.type_name = IDExpression(match.group('name'))

    def __str__(self):
        return f'{self.class_key} {self.type_name}'


class TypeSpecifier(Node):
    producers = (FundamentalTypeSpecifier, ElaboratedTypeSpecifier)
    
    @lazyattr
    def regex(cls):
        return re.compile('|'.join(f'{p.genericPattern}' for p in cls.producers), re.VERBOSE)


class TypeID(Node):
    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            (?:(?P<lhCVQuals>{CVQualifiers.genericPattern})\s+)?
            (?P<typeSpecifier>{TypeSpecifier.genericPattern})
            (?:\s+(?P<rhCVQuals>{CVQualifiers.genericPattern}))?
            (?:\s*(?P<declarator>{AbstractDeclarator.genericPattern}))?
        ''', re.VERBOSE)

    def build(self, match):
        self.type_spec = TypeSpecifier(match.group('typeSpecifier'))
        cvQuals = f'{match.group('lhCVQuals') or ''} {match.group('rhCVQuals') or ''}'.strip()
        self.cv_qualifiers = CVQualifiers(cvQuals) if cvQuals else None
        self.declarator = (
            AbstractDeclarator(declarator) 
            if (declarator := match.group('declarator'))
            else None
        )
        
        if self.cv_qualifiers is not None and self.isPtr():
            if Keys.CONST in self.cv_qualifiers:
                self.declarator.isPtrToConst = True
            if Keys.VOLATILE in self.cv_qualifiers:
                self.declarator.isPtrToVolatile = True
    
    def __str__(self):
        return (
            f"{self.type_spec} {f'{self.cv_qualifiers} ' if self.cv_qualifiers else ''}{self.declarator or ''}".strip()
        )
    
    def isElaborated(self):
        return isinstance(self.type_spec, ElaboratedTypeSpecifier)
    
    def isFundamental(self):
        return isinstance(self.type_spec, FundamentalTypeSpecifier)
    
    def isPtr(self):
        return isinstance(self.declarator, PtrAbstractDeclarator)


class TemplateArgument(Node):
    producers = (TypeID, ConstantExpression)
    
    @lazyattr
    def regex(cls):
        return re.compile(rf'{TypeID.genericPattern}|{ConstantExpression.genericPattern}', re.VERBOSE)


class ConstructorID(Node):
    @lazyattr
    def regex(cls):
        return re.compile(rf'''
                (?P<scope>{NestedNameSpecifier.genericPattern})
                (?P<name>{UnqualifiedID.genericPattern})
            ''', re.VERBOSE)

    def build(self, match):
        self.scope = NestedNameSpecifier(match.group('scope'))
        self.identifier = UnqualifiedID(match.group('name'))
        
        if self.identifier != self.scope.identifier:
            return self.BUILD_ERROR
    
    def __str__(self):
        return f'{self.scope}{self.identifier}'


class DestructorID(Node):
    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            (?P<scope>{NestedNameSpecifier.genericPattern})
            {Keys.DESTRUCTOR}(?P<name>{UnqualifiedID.genericPattern})
        ''', re.VERBOSE)
    
    def build(self, match):
        self.scope = NestedNameSpecifier(match.group('scope'))
        self.identifier = UnqualifiedID(match.group('name'))
        
        if self.identifier != self.scope.identifier:
            return self.BUILD_ERROR
    
    def __str__(self):
        return f'{self.scope}{Keys.DESTRUCTOR}{self.identifier}'


class ImplicitPropertyID(Node):
    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            (?P<scope>{NestedNameSpecifier.genericPattern})
            (?P<name>`{Keys.VFTABLE}'|{Keys.VFTABLE})
        ''', re.VERBOSE)
    
    def build(self, match):
        self.scope = NestedNameSpecifier(match.group('scope'))
        self.identifier = Keys.VFTABLE
    
    def __str__(self):
        return f"{self.scope}`{self.identifier}'"


class QualifiedID(Node):
    producers = (OperatorFunctionID, DestructorID, ConstructorID, ImplicitPropertyID)
    
    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            (?P<scope>{NestedNameSpecifier.genericPattern})
            (?P<name>{UnqualifiedID.genericPattern})
        ''', re.VERBOSE)
    
    def build(self, match):
        self.scope = NestedNameSpecifier(match.group('scope'))
        self.identifier = UnqualifiedID(match.group('name'))
    
    def __str__(self):
        return f'{self.scope}{self.identifier}'


class IDExpression(Node):
    producers = (UnqualifiedID, QualifiedID)

    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            (?:{NestedNameSpecifier.genericPattern})?
            (?:{Keys.DESTRUCTOR})?
            {UnqualifiedID.genericPattern}
        ''', re.VERBOSE)


class PtrExtendedQualifier(Node):
    @lazyattr
    def regex(cls):
        return re.compile(rf'{Keys.PTR64}|{Keys.PTR32}', re.VERBOSE)

    def build(self, match):
        self.qualifier = match.group()
    
    @lazyattr
    def PTR64(cls):
        return cls(Keys.PTR64)
    
    def __str__(self):
        return self.qualifier


class PtrExtQualifiers(Node):
    @lazyattr
    def regex(cls):
        return re.compile(rf'(?:({PtrExtendedQualifier.genericPattern})\s*)*', re.VERBOSE)
    
    def build(self, match):
        if len(set(match.groups())) < len(match.groups()):
            return self.BUILD_ERROR
        
        self.qualifiers = [PtrExtendedQualifier(qual) for qual in match.groups() if qual]

    def __iter__(self):
        for i in range(len(self.qualifiers)) :
            yield self.qualifiers[i]
    
    def __getitem__(self, key):
        return self.qualifiers[key]
    
    def __str__(self):
        return ' '.join(map(str, self)).strip()


class PtrOperator(Node):
    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            (?P<ptrToMember>{NestedNameSpecifier.genericPattern})?
            (?P<operator>{Keys.RVAL_REF}|{Keys.REF}|{re.escape(Keys.PTR)})\s*
            (?:(?P<cvQuals>{CVQualifiers.genericPattern})\s*)?
            (?P<ptrExtQuals>{PtrExtQualifiers.genericPattern})?
        ''', re.VERBOSE)

    @lazyattr
    def PTR(cls):
        return cls(Keys.PTR)

    @lazyattr
    def PTR_CONST(cls):
        return cls(f'{Keys.PTR} {Keys.CONST}')

    @lazyattr
    def PTR_VOLATILE(cls):
        return cls(f'{Keys.PTR} {Keys.VOLATILE}')

    @lazyattr
    def PTR_CONST_VOLATILE(cls):
        return cls(f'{Keys.PTR} {Keys.CONST} {Keys.VOLATILE}')

    @lazyattr
    def REF(cls):
        return cls(Keys.REF)

    @lazyattr
    def RVAL_REF(cls):
        return cls(Keys.RVAL_REF)

    def build(self, match):
        self.ptr_to_member = (
            NestedNameSpecifier(match.group('ptrToMember')) 
            if match.group('ptrToMember') 
            else None
        )
        self.operator = match.group('operator')
        self.cv_qualifiers = (
            CVQualifiers(cvQuals) 
            if (cvQuals := match.group('cvQuals')) and self.operator == Keys.PTR
            else None
        )
        self.ext_qualifiers = PtrExtQualifiers(match.group('ptrExtQuals') or Keys.PTR64) 

    def __str__(self):
        return (
            f"{self.ptr_to_member or ''}{self.operator} {self.ext_qualifiers} {self.cv_qualifiers or ''}".strip()
        )
    
    def isPtrToMember(self):
        return self.ptr_to_member is not None


class PtrAbstractDeclarator(Node):
    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            (?P<ptrOp>{PtrOperator.genericPattern})\s*
            (?P<ptrOpSeq>(?:{PtrOperator.genericPattern}\s*)*)
            (?P<noPtr>{NoPtrAbstractDeclarator.genericPattern})?
        ''', re.VERBOSE)
    
    def build(self, match):
        self.operator = PtrOperator(match.group('ptrOp'))
        self.prev = (
            PtrAbstractDeclarator(ptrOpSeq) 
            if (ptrOpSeq := match.group('ptrOpSeq')) 
            else None
        )
        self.no_ptr = (
            NoPtrAbstractDeclarator(noPtr) 
            if (noPtr := match.group('noPtr'))
            else None
        )
        
        self.isPtrToConst = False
        self.isPtrToVolatile = False
        
        if self.prev is not None and self.prev.operator.cv_qualifiers is not None:
            if CVQualifiers.CONST in self.prev.operator.cv_qualifiers:
                self.isPtrToConst = True
            if CVQualifiers.VOLATILE in self.prev.operator.cv_qualifiers:
                self.isPtrToVolatile = True

    def __str__(self):
        return f"{self.prev or ''} {self.operator}{self.no_ptr or ''}".strip()


class ParametersDeclarator(Node):
    @lazyattr
    def regex(cls):
        return re.compile(rf'{re.escape(Keys.L_PAREN)}.*{re.escape(Keys.R_PAREN)}', re.VERBOSE)
    
    @staticmethod
    def split_parameters(params_list: str):
        if not params_list:
            return []
        result = []
        buffer = ""
        depth = 0
        
        for char in params_list[1:-1]:
            if char in (Keys.L_ANG_BRACKET, Keys.L_PAREN):
                depth += 1
            elif char in (Keys.R_ANG_BRACKET, Keys.R_PAREN):
                depth = max(depth - 1, 0)
            elif char == Keys.SEPARATOR and depth == 0:
                result.append(buffer.strip())
                buffer = ""
                continue

            buffer += char

        if buffer:
            result.append(buffer.strip())

        return result
    
    @lazyattr
    def VOID(cls):
        return cls(f'{Keys.L_PAREN}{Keys.VOID}{Keys.R_PAREN}')
    
    def build(self, match):
        params = self.split_parameters(match.group())
        self.params_list = [TypeID(param) for param in params or [Keys.VOID]]
    
    def __getitem__(self, key):
        return self.params_list[key]
    
    def __str__(self):
        return (
            f'{Keys.L_PAREN}'
            f'{Keys.SEPARATOR.join([str(param) for param in self.params_list]).strip()}'
            f'{Keys.R_PAREN}'
        )


class FuncAbstractDeclarator(Node):
    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            (?:(?P<callConv>{CallConvention.genericPattern})\s*)?
            (?P<params>{ParametersDeclarator.genericPattern})
        ''', re.VERBOSE)
    
    def build(self, match):
        self.call_conv = (
            CallConvention(callConv) 
            if (callConv := match.group('callConv'))
            else CallConvention.CDECL
        )
        self.params = ParametersDeclarator(match.group('params'))

    def __str__(self):
        return f'{self.call_conv}{self.params}'


class FuncPtrAbstractDeclarator(Node):
    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            {re.escape(Keys.L_PAREN)}
            (?:(?P<callConv>{CallConvention.genericPattern})\s*)?
            (?P<ptrOps>(?:{PtrOperator.genericPattern}\s*)+)
            {re.escape(Keys.R_PAREN)}
            (?P<params>{ParametersDeclarator.genericPattern})
        ''', re.VERBOSE)
    
    def build(self, match):
        self.call_conv = (
            CallConvention(callConv) 
            if (callConv := match.group('callConv'))
            else CallConvention.CDECL
        )
        self.ptr_declarator = PtrAbstractDeclarator(match.group('ptrOps'))
        self.params = ParametersDeclarator(match.group('params'))

    def __str__(self):
        return f'({self.call_conv}{self.ptr_declarator}){self.params}'    


class NoPtrAbstractDeclarator(Node):
    producers = (FuncPtrAbstractDeclarator, FuncAbstractDeclarator)
    
    @lazyattr
    def regex(cls):
        return re.compile('|'.join(f'{p.genericPattern}' for p in cls.producers), re.VERBOSE)


class AbstractDeclarator(Node):
    producers = (PtrAbstractDeclarator, NoPtrAbstractDeclarator)

    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            (?:{PtrOperator.genericPattern}\s*(?:{PtrOperator.genericPattern}\s*)*)?
            (?:{NoPtrAbstractDeclarator.genericPattern})?
        ''', re.VERBOSE)


class FunctionClass(Node):
    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            (?:
                (?P<access>{AccessSpecifier.genericPattern})\s*
                (?:\{Keys.ACCESS_SCOPE}\s*)?
                (?P<resolution>{ResolutionSpecifier.genericPattern})?
            )?
        ''', re.VERBOSE)
    
    @lazyattr
    def GLOBAL(cls):
        return cls('')
    
    @lazyattr
    def PRIVATE(cls):
        return cls(Keys.PRIVATE)
    
    @lazyattr
    def PROTECTED(cls):
        return cls(Keys.PROTECTED)
    
    @lazyattr
    def PUBLIC(cls):
        return cls(Keys.PUBLIC)
    
    @lazyattr
    def PRIVATE_STATIC(cls):
        return cls(f'{Keys.PRIVATE} {Keys.STATIC}')
    
    @lazyattr
    def PROTECTED_STATIC(cls):
        return cls(f'{Keys.PROTECTED} {Keys.STATIC}')
    
    @lazyattr
    def PUBLIC_STATIC(cls):
        return cls(f'{Keys.PUBLIC} {Keys.STATIC}')
    
    @lazyattr
    def PRIVATE_VIRTUAL(cls):
        return cls(f'{Keys.PRIVATE} {Keys.VIRTUAL}')
    
    @lazyattr
    def PROTECTED_VIRTUAL(cls):
        return cls(f'{Keys.PROTECTED} {Keys.VIRTUAL}')
    
    @lazyattr
    def PUBLIC_VIRTUAL(cls):
        return cls(f'{Keys.PUBLIC} {Keys.VIRTUAL}')
    
    def build(self, match):
        access = match.group('access')
        resolution = match.group('resolution')
        if access is None and resolution is not None:
            return self.BUILD_ERROR # can't have resolution spec without access spec
        self.access = AccessSpecifier(access) if access is not None else None
        self.resolution = ResolutionSpecifier(resolution) if resolution is not None else None
    
    def __str__(self):
        if self.access is None:
            return ''
        return f'{self.access}: {self.resolution or ""}'.strip()


class FuncNode(Node):
    def __str__(self):
        return (
            f"{self._class if self._class is not FunctionClass.GLOBAL else ''} "
            f"{self.return_type or '\b'} {self.call_conv} "
            f"{self.identifier}{self.params}"
            f"{getattr(self, 'instance_cv_quals', None) or ''} "
            f"{getattr(self, 'instance_ext_quals', None) or ''}"
        ).strip()
    
    def isGlobal(self):
        return self._class == FunctionClass.GLOBAL
  
    def isStatic(self):
        return self._class.resolution == ResolutionSpecifier.STATIC


class ConstructorDeclaration(FuncNode):
    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            (?P<funcClass>
                (?:{AccessSpecifier.genericPattern})\s*
                (?:\{Keys.ACCESS_SCOPE}\s*)?
            )\s+
            (?!(?P<retType>{TypeID.genericPattern})\s+)
            (?:(?P<callConv>{CallConvention.genericPattern})\s+)?
            (?P<identifier>{ConstructorID.genericPattern})\s*
            (?P<params>{ParametersDeclarator.genericPattern})
            (?:\s*(?P<instExtQuals>{PtrExtQualifiers.genericPattern}))?
        ''', re.VERBOSE)

    def build(self, match):
        self._class = FunctionClass(match.group('funcClass'))
        self.return_type =  None
        self.call_conv = (
            CallConvention(callConv) 
            if (callConv := match.group('callConv'))
            else CallConvention.CDECL
        )
        self.identifier = ConstructorID(match.group('identifier'))
        if self.identifier is None:
            return self.BUILD_ERROR
        self.params = ParametersDeclarator(match.group('params'))
        self.instance_cv_quals = None
        self.instance_ext_quals = PtrExtQualifiers(match.group('instExtQuals') or Keys.PTR64)


class DestructorDeclaration(FuncNode):
    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            (?P<funcClass>
                (?:{AccessSpecifier.genericPattern})\s*
                (?:\{Keys.ACCESS_SCOPE}\s*)?
                (?:{Keys.VIRTUAL})?
            )\s+
            (?!(?P<retType>{TypeID.genericPattern})\s+)
            (?:(?P<callConv>{CallConvention.genericPattern})\s+)?
            (?P<identifier>{DestructorID.genericPattern})\s*
            (?P<params>{ParametersDeclarator.genericPattern})
            (?!\s*(?P<instCVQuals>{CVQualifiers.genericPattern}))?
            (?:\s*(?P<instExtQuals>{PtrExtQualifiers.genericPattern}))?
        ''', re.VERBOSE)

    def build(self, match):
        self._class = FunctionClass(match.group('funcClass'))
        self.return_type =  None
        self.call_conv = (
            CallConvention(callConv) 
            if (callConv := match.group('callConv'))
            else CallConvention.CDECL
        )
        self.identifier = DestructorID(match.group('identifier'))
        self.params = ParametersDeclarator(match.group('params'))
        self.instance_cv_quals = None
        self.instance_ext_quals = PtrExtQualifiers(match.group('instExtQuals') or Keys.PTR64)

        if (
            self.identifier.scope.identifier != self.identifier.identifier
            or
            self.params != ParametersDeclarator.VOID
        ):
            return self.BUILD_ERROR


class OperatorMethodDeclaration(FuncNode):
    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            (?P<funcClass>
                (?:{AccessSpecifier.genericPattern})\s*
                (?:\{Keys.ACCESS_SCOPE}\s*)?
            )\s+
            (?P<retType>{TypeID.genericPattern})\s+
            (?:(?P<callConv>{CallConvention.genericPattern})\s+)?
            (?P<identifier>{OperatorFunctionID.genericPattern})\s*
            (?P<params>{ParametersDeclarator.genericPattern})
            (?:\s*(?P<instCVQuals>{CVQualifiers.genericPattern}))?
            (?:\s*(?P<instExtQuals>{PtrExtQualifiers.genericPattern}))?
        ''', re.VERBOSE)

    def build(self, match):
        self._class = FunctionClass(match.group('funcClass'))
        self.return_type = TypeID(match.group('retType'))
        self.call_conv = (
            CallConvention(callConv) 
            if (callConv := match.group('callConv'))
            else CallConvention.CDECL
        )
        self.identifier = OperatorFunctionID(match.group('identifier'))
        self.params = ParametersDeclarator(match.group('params'))
        self.instance_cv_quals = (
            CVQualifiers(instCVQuals) 
            if (instCVQuals := match.group('instCVQuals'))
            else None
        )
        self.instance_ext_quals = PtrExtQualifiers(match.group('instExtQuals') or Keys.PTR64)


class MethodDeclaration(FuncNode):
    producers = (OperatorMethodDeclaration, DestructorDeclaration, ConstructorDeclaration)
    
    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            (?P<funcClass>
                (?:{AccessSpecifier.genericPattern})\s*
                (?:\{Keys.ACCESS_SCOPE}\s*)?
                (?:{ResolutionSpecifier.genericPattern})?
            )\s+
            (?P<retType>{TypeID.genericPattern})\s+
            (?:(?P<callConv>{CallConvention.genericPattern})\s+)?
            (?P<identifier>{QualifiedID.genericPattern})\s*
            (?P<params>{ParametersDeclarator.genericPattern})
            (?:\s*(?P<instCVQuals>{CVQualifiers.genericPattern}))?
            (?:\s*(?P<instExtQuals>{PtrExtQualifiers.genericPattern}))?
        ''', re.VERBOSE)
    
    def build(self, match):
        self._class = FunctionClass(match.group('funcClass'))
        self.return_type = TypeID(match.group('retType'))
        self.call_conv = (
            CallConvention(callConv) 
            if (callConv := match.group('callConv')) 
            else CallConvention.CDECL
        )
        self.identifier = QualifiedID(match.group('identifier'))
        self.params = ParametersDeclarator(match.group('params'))
        self.instance_cv_quals = (
            CVQualifiers(instCVQuals) 
            if (instCVQuals := match.group('instCVQuals'))
            else None
        )
        self.instance_ext_quals = PtrExtQualifiers(match.group('instExtQuals') or Keys.PTR64)
        
        if self.isStatic() and self.instance_quals is not None:
            return self.BUILD_ERROR # static method cannot have instance qualifiers


class FunctionDeclaration(FuncNode):
    producers = (MethodDeclaration,)
    
    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            (?P<retType>{TypeID.genericPattern})\s+
            (?:(?P<callConv>{CallConvention.genericPattern})\s+)?
            (?P<identifier>{IDExpression.genericPattern})\s*
            (?P<params>{ParametersDeclarator.genericPattern})
        ''', re.VERBOSE)
    
    def build(self, match):
        self._class = FunctionClass.GLOBAL
        self.return_type = TypeID(match.group('retType'))
        self.call_conv = (
            CallConvention(callConv) 
            if (callConv := match.group('callConv'))
            else CallConvention.CDECL
        )
        self.identifier = IDExpression(match.group('identifier'))
        self.params = ParametersDeclarator(match.group('params'))


class VariableClass(Node):
    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            (?:
                (?P<access>{AccessSpecifier.genericPattern})\s*
                (?:\{Keys.ACCESS_SCOPE}\s*)?
                (?P<resolution>{Keys.STATIC})?
            )?
        ''', re.VERBOSE)
    
    @lazyattr
    def GLOBAL(cls):
        return cls('')
    
    @lazyattr
    def PRIVATE_STATIC(cls):
        return cls(f'{Keys.PRIVATE} {Keys.STATIC}')
    
    @lazyattr
    def PROTECTED_STATIC(cls):
        return cls(f'{Keys.PROTECTED} {Keys.STATIC}')
    
    @lazyattr
    def PUBLIC_STATIC(cls):
        return cls(f'{Keys.PUBLIC} {Keys.STATIC}')
 
    def build(self, match):
        access = match.group('access')
        resolution = match.group('resolution')
        if access is None and resolution is not None:
            return Node.BUILD_ERROR
        self.access = AccessSpecifier(access) if access is not None else None
        self.resolution = ResolutionSpecifier(resolution) if resolution is not None else None
    
    def __str__(self):
        if self.access is None:
            return ''
        return f'{self.access}: {self.resolution or ""}'.strip()


class VarNode(Node):
    def __str__(self):
        return (
            f'{self._class if self._class and self._class is not VariableClass.GLOBAL else ''} '
            f'{self.decl_type or ''} {self.identifier}'
        ).strip()
    
    def isGlobal(self):
        return self._class == VariableClass.GLOBAL
  
    def isStatic(self):
        return self._class.resolution == ResolutionSpecifier.STATIC


class ImplicitPropertyDeclaration(VarNode):
    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            (?:{Keys.CONST}\s+)?
            (?P<identifier>{ImplicitPropertyID.genericPattern})
        ''', re.VERBOSE)
        
    def build(self, match):
        self._class = None
        self.decl_type = None
        self.identifier = ImplicitPropertyID(match.group('identifier'))
        self.storage_quals = CVQualifiers.CONST
    
    def __str__(self):
        return f'{Keys.CONST} {self.identifier}'


class PropertyDeclaration(VarNode):
    producers = (ImplicitPropertyDeclaration,)
    
    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            (?P<varClass>{VariableClass.genericPattern})\s+
            (?P<declType>{TypeID.genericPattern})\s*
            (?P<identifier>
                {NestedNameSpecifier.genericPattern}
                {Identifier.genericPattern}
            )
        ''', re.VERBOSE)
    
    def build(self, match):
        self._class = VariableClass(match.group('varClass'))
        self.decl_type = TypeID(match.group('declType'))
        self.identifier = QualifiedID(match.group('identifier'))
        self.storage_quals = (
            self.decl_type.declarator.operator.cv_qualifiers 
            if self.decl_type.declarator and self.decl_type.declarator.isPtr()
            else self.decl_type.cv_qualifiers
        )


class VariableDeclaration(VarNode):
    producers = (PropertyDeclaration,)
    
    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            (?P<declType>{TypeID.genericPattern})\s*
            (?P<identifier>
                (?:{NestedNameSpecifier.genericPattern})?
                {Identifier.genericPattern}
            )
        ''', re.VERBOSE)
    
    def build(self, match):
        self._class = VariableClass.GLOBAL
        self.decl_type = TypeID(match.group('declType'))
        self.storage_quals = (
            self.decl_type.declarator.operator.cv_qualifiers 
            if self.decl_type.declarator and self.decl_type.declarator.isPtr()
            else self.decl_type.cv_qualifiers
        )
        self.identifier = IDExpression(match.group('identifier')) 


class Declaration(Node):
    producers = (FunctionDeclaration, VariableDeclaration)
    
    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            (?:{Keys.CONST}\s+)?
            (?:{AccessSpecifier.genericPattern}\s*)?
            (?:\{Keys.ACCESS_SCOPE}\s*)?
            (?:{ResolutionSpecifier.genericPattern})?
            {TypeID.genericPattern}\s+
            (?:{CallConvention.genericPattern}\s+)?
            {IDExpression.genericPattern}\s*
            (?:{ParametersDeclarator.genericPattern})?
            (?:\s*{CVQualifiers.genericPattern})?
            (?:\s*{PtrExtQualifiers.genericPattern})?
        ''', re.VERBOSE)


class Mangler:
    def __init__(self, decl=None):
        self.result = ''
        self.name_back_refs = []
        if not decl:
            return
        
        self.original = Declaration(decl) if not isinstance(decl, Declaration) else decl
        self.result = self.mangle(self.original)
    
    def mangleFunctionClass(self, func_cls: FunctionClass):
        match func_cls:
            case FunctionClass.PRIVATE:
                return 'A'
            case FunctionClass.PRIVATE_STATIC:
                return 'C'
            case FunctionClass.PRIVATE_VIRTUAL:
                return 'E'
            case FunctionClass.PROTECTED:
                return 'I'
            case FunctionClass.PROTECTED_STATIC:
                return 'K'
            case FunctionClass.PROTECTED_VIRTUAL:
                return 'M'
            case FunctionClass.PUBLIC:
                return 'Q'
            case FunctionClass.PUBLIC_STATIC:
                return 'S'
            case FunctionClass.PUBLIC_VIRTUAL:
                return 'U'
            case FunctionClass.GLOBAL:
                return 'Y'

    def mangleUnqualifiedID(self, _id: UnqualifiedID):
        if _id in self.name_back_refs:
            return str(self.name_back_refs.index(_id))

        match _id:
            case TemplateID():
                return self.mangleTemplateID(_id)
            case _:
                if len(self.name_back_refs) < 10:
                    self.name_back_refs.append(_id)
                return str(_id) + '@'
    
    def mangleTemplateID(self, template_id: TemplateID):
        result = '?$'
        mangler = Mangler()
        result += mangler.mangleUnqualifiedID(template_id.identifier)
        for arg in template_id.template_args_list:
            result += mangler.mangleTemplateArg(arg)
        result += '@'
        return result
    
    def mangleScope(self, scope: NestedNameSpecifier):
        result = self.mangleUnqualifiedID(scope.identifier)
        if scope.scope:
            result += self.mangleScope(scope.scope)
        return result
    
    def mangleID(self, _id: IDExpression):
        match _id:
            case ConstructorID():
                result = '?0'
            case DestructorID():
                result = '?1'
            case OperatorFunctionID():
                match _id.identifier:
                    case OverloadableOperator.NEW:
                        result = '?2'
                    case OverloadableOperator.ASSIGN:
                        result = '?4'
                    case OverloadableOperator.RSHIFT:
                        result = '?5'
                    case OverloadableOperator.LSHIFT:
                        result = '?6'
                    case OverloadableOperator.NOT:
                        result = '?7'
                    case OverloadableOperator.EQUALS:
                        result = '?8'
                    case OverloadableOperator.NOT_EQUALS:
                        result = '?9'
                    case OverloadableOperator.ARROW:
                        result = '?C'
                    case OverloadableOperator.PTR:
                        result = '?D'
                    case OverloadableOperator.INCREMENT:
                        result = '?E'
                    case OverloadableOperator.DECREMENT:
                        result = '?F'
                    case OverloadableOperator.SUBTRACT:
                        result = '?G'
                    case OverloadableOperator.ADD:
                        result = '?H'
                    case OverloadableOperator.PTR_TO_MEMBER:
                        result = '?J'
                    case OverloadableOperator.DIVIDE:
                        result = '?K'
                    case OverloadableOperator.MODULUS:
                        result = '?L'
                    case OverloadableOperator.LESS_THAN:
                        result = '?M'
                    case OverloadableOperator.LESS_OR_EQUALS:
                        result = '?N'
                    case OverloadableOperator.GREATER_THAN:
                        result = '?O'
                    case OverloadableOperator.GREATER_OR_EQUALS:
                        result = '?P'
                    case OverloadableOperator.COMMA:
                        result = '?Q'
                    case OverloadableOperator.MULT_ASSIGN:
                        result = '?X'
                    case OverloadableOperator.ADD_ASSIGN:
                        result = '?Y'
                    case OverloadableOperator.SUBTRACT_ASSIGN:
                        result = '?Z'
                    case OverloadableOperator.DIVIDE_ASSIGN:
                        result = '?_0'
            case ImplicitPropertyID():
                result = '?_7'
            case UnqualifiedID():
                result = self.mangleUnqualifiedID(_id)
            case _:
                result = self.mangleUnqualifiedID(_id.identifier)

        if (scope := getattr(_id, 'scope', None)) is not None:
            result += self.mangleScope(scope)
        return result + '@'

    def mangleFundamentalType(self, fundamental_type: FundamentalTypeSpecifier):
        match fundamental_type:
            case FundamentalTypeSpecifier.VOID:
                return 'X'
            case FundamentalTypeSpecifier.SCHAR:
                return 'C'
            case FundamentalTypeSpecifier.CHAR:
                return 'D'
            case FundamentalTypeSpecifier.UCHAR:
                return 'E'
            case FundamentalTypeSpecifier.SHORT:
                return 'F'
            case FundamentalTypeSpecifier.USHORT:
                return 'G'
            case FundamentalTypeSpecifier.INT:
                return 'H'
            case FundamentalTypeSpecifier.UINT:
                return 'I'
            case FundamentalTypeSpecifier.LONG:
                return 'J'
            case FundamentalTypeSpecifier.ULONG:
                return 'K'
            case FundamentalTypeSpecifier.FLOAT:
                return 'M'
            case FundamentalTypeSpecifier.DOUBLE:
                return 'N'
            case FundamentalTypeSpecifier.INT64:
                return '_J'
            case FundamentalTypeSpecifier.UINT64:
                return '_K'
            case FundamentalTypeSpecifier.BOOL:
                return '_N'
            case FundamentalTypeSpecifier.WCHAR:
                return '_W'

    def mangleClassKey(self, cls_key: ClassKey):
        match cls_key:
            case ClassKey.CLASS:
                return 'V'
            case ClassKey.STRUCT:
                return 'U'

    def mangleElaboratedType(self, elaborated_type: ElaboratedTypeSpecifier):
        result = self.mangleClassKey(elaborated_type.class_key)
        result += self.mangleID(elaborated_type.type_name)
        return result

    def mangleCVQualifiers(self, cv_quals: CVQualifiers):
        match cv_quals:
            case None:
                return 'A'
            case CVQualifiers.CONST:
                return 'B'
            case CVQualifiers.VOLATILE:
                return 'C'
            case CVQualifiers.CONST_VOLATILE:
                return 'D'

    def mangleCallConvention(self, call_conv: CallConvention):
        match call_conv:
            case CallConvention.CDECL:
                return 'A'
            case CallConvention.STDCALL:
                return 'G'
            case CallConvention.FASTCALL:
                return 'I'

    def mangleTypeSpec(self, type_spec: TypeSpecifier):
        match type_spec:
            case FundamentalTypeSpecifier():
                return self.mangleFundamentalType(type_spec)
            case ElaboratedTypeSpecifier():
                return self.mangleElaboratedType(type_spec)
    
    def manglePtrExtQualifier(self, ptr_ext_qual: PtrExtendedQualifier):
        match ptr_ext_qual:
            case Keys.PTR64:
                return 'E'
    
    def manglePtrExtQualifiers(self, ptr_ext_quals: PtrExtQualifiers):
        return ''.join(self.manglePtrExtQualifier(qual) for qual in ptr_ext_quals)
    
    def manglePtrCVQualifiers(self, ptr_op: PtrOperator):
        match ptr_op:
            case PtrOperator.PTR:
                return 'P'
            case PtrOperator.PTR_CONST:
                return 'Q'
            case PtrOperator.PTR_VOLATILE:
                return 'R'
            case PtrOperator.PTR_CONST_VOLATILE:
                return 'S'
            case PtrOperator.REF:
                return 'A'
            case PtrOperator.RVAL_REF:
                return '$$Q'
    
    def manglePtrDeclarator(self, ptr_decl: PtrAbstractDeclarator):
        result = self.manglePtrCVQualifiers(ptr_decl.operator)
        result += self.manglePtrExtQualifiers(ptr_decl.operator.ext_qualifiers)
        result += self.mangleCVQualifiers(
            CVQualifiers.CONST_VOLATILE if ptr_decl.isPtrToConst and ptr_decl.isPtrToVolatile
            else 
                CVQualifiers.CONST if ptr_decl.isPtrToConst
            else
                CVQualifiers.VOLATILE if ptr_decl.isPtrToVolatile
            else
                None
        )
        
        if ptr_decl.prev:
            result += self.manglePtrDeclarator(ptr_decl.prev)
        return result
    
    def mangleFuncDeclarator(self, func_decl: FuncAbstractDeclarator | FuncPtrAbstractDeclarator):
        result = ''
        if isinstance(func_decl, FuncPtrAbstractDeclarator):
            result += self.manglePtrCVQualifiers(func_decl.ptr_declarator.operator)
            if func_decl.ptr_declarator.operator.isPtrToMember():
                result += '8'
            else:
                result += '6'
        else:
            result += '$$A6'
        
        return result
    
    def mangleNoPtrDeclarator(self, no_ptr_decl: NoPtrAbstractDeclarator):
        return self.mangleFuncDeclarator(no_ptr_decl)
    
    def mangleDeclarator(self, declarator: AbstractDeclarator):
        match declarator:
            case PtrAbstractDeclarator():
                return self.manglePtrDeclarator(declarator)
            case NoPtrAbstractDeclarator():
                return self.mangleNoPtrDeclarator(declarator)
    
    def mangleTypeID(self, type_id: TypeID):
        if not type_id:
            return '@'
        
        result = ''
        if decl := type_id.declarator:
            result += self.mangleDeclarator(decl)
            if (
                isinstance(decl, FuncAbstractDeclarator) 
                or 
                isinstance(decl, FuncPtrAbstractDeclarator)
            ):
                ret_type = TypeID(f"{type_id.type_spec} {type_id.cv_qualifiers or ''}".rstrip())
                result += self.mangleFunctionType(decl.call_conv, ret_type, decl.params)
                return result
        
        result += self.mangleTypeSpec(type_id.type_spec)
        return result
    
    def mangleConstExpression(self, const_expr: ConstantExpression):
        result = '$0'
        value = int(const_expr.value)
        if value == 0:
            return result + "A@"
        elif 1 <= value <= 10:
            return result + str(value - 1)
        else:
            encoded = []
            while value != 0:
                nibble = value & 0xf
                encoded.append(chr(ord('A') + nibble))
                value >>= 4
            encoded.reverse()
            return result + ''.join(encoded) + '@'
    
    def mangleTemplateArg(self, template_arg: TemplateArgument):
        match template_arg:
            case TypeID():
                return self.mangleTypeID(template_arg)
            case ConstantExpression():
                return self.mangleConstExpression(template_arg)
    
    def mangleFunctionType(self, call_conv, ret_type, params):
        result = self.mangleCallConvention(call_conv)

        if ret_type and ret_type.isElaborated() and not ret_type.isPtr():
            result += '?' + self.mangleCVQualifiers(ret_type.cv_qualifiers)
        result += self.mangleTypeID(ret_type)
        
        param_back_refs = []
        for param in params:
            if param in param_back_refs:
                result += str(param_back_refs.index(param))
            else:
                if len(param_back_refs) < 10:
                    param_back_refs.append(param)
                result += self.mangleTypeID(param)
        
        if params != ParametersDeclarator.VOID:
            result += '@'

        result += 'Z'
        return result
    
    def mangleFunctionDeclaration(self, func: FunctionDeclaration):
        result = self.mangleID(func.identifier)
        result += self.mangleFunctionClass(func._class)

        if not (func.isStatic() or func.isGlobal()):
            result += self.manglePtrExtQualifiers(func.instance_ext_quals)
            result += self.mangleCVQualifiers(func.instance_cv_quals)
        
        result += self.mangleFunctionType(func.call_conv, func.return_type, func.params)
        return result

    def mangleVariableClass(self, var_cls: VariableClass):
        match var_cls:
            case VariableClass.PRIVATE_STATIC:
                return '0'
            case VariableClass.PROTECTED_STATIC:
                return '1'
            case VariableClass.PUBLIC_STATIC:
                return '2'
            case VariableClass.GLOBAL:
                return '3'
            case None:
                return '6'
    
    def mangleVariableDeclaration(self, var_def: VariableDeclaration):
        result = self.mangleID(var_def.identifier)
        result += self.mangleVariableClass(var_def._class)

        if var_def.decl_type:
            result += self.mangleTypeID(var_def.decl_type)
            result += 'E'
        
        result += self.mangleCVQualifiers(var_def.storage_quals)

        if var_def.decl_type is None:
            result += '@'
        
        return result
    
    def mangle(self, decl: Declaration):
        result = '?'
        match decl:
            case FunctionDeclaration():
                result += self.mangleFunctionDeclaration(decl)
            case VariableDeclaration():
                result += self.mangleVariableDeclaration(decl)
        
        return result
    
    def __str__(self):
        return self.result


arg_parser = argparse.ArgumentParser(description="Accepts C++ declarations for mangling and hashing")
arg_parser.add_argument("declarations", nargs='+', help='One or more quote encased C++ declarations, e.g. "public: void MyClass::myMethod(void*) const"')

def mangle_decls(decl_str_list):
    declarations = list(Declaration(decl_str) for decl_str in decl_str_list)
    return dict((decl, str(Mangler(decl))) for decl in declarations)

def main(argv=None):
    args = arg_parser.parse_args(argv)
    return mangle_decls(args.declarations)

if __name__ == "__main__":
    result = main()
    for (decl, mangled) in result.items():
        if mangled:
            print(f"\nMangling of :- \"{decl}\"\nis :- \"{mangled}\"")
