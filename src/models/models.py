from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()


class FactProgAcad(Base):
    __tablename__ = 'fact_prog_acad'

    PERIODO = Column(Integer, primary_key=True)
    SEDE = Column(String, primary_key=True)
    MODALIDAD = Column(String)
    FASE = Column(String)
    NIVEL = Column(String)
    CODIGO_DE_CURSO = Column(String, primary_key=True)
    DESCRIPCION = Column(String)
    FRECUENCIA = Column(String)
    INTENSIDAD = Column(String)
    HORARIO = Column(String, primary_key=True)
    AULA = Column(String, primary_key=True)
    COD_DOCENTE = Column(String, nullable=True)
    PROFESOR = Column(String, nullable=True)
    CANT_MATRICULADOS = Column(Integer)
    VACANTES_USADAS = Column(Integer)
    VACANTES_DISP = Column(Integer)
    VAC_HABILITADAS = Column(Integer)
    AFORO = Column(Integer)
    ESTADO = Column(String)
    FECHA_CREACION = Column(String)
    created_at = Column(DateTime, server_default=func.now())


class FactPredict(Base):
    __tablename__ = 'fact_predict'

    PERIODO = Column(Integer, primary_key=True)
    SEDE = Column(String, primary_key=True)
    CODIGO_DE_CURSO = Column(String, primary_key=True)
    HORARIO = Column(String, primary_key=True)
    FORECAST_AULAS = Column(Integer)
    FORECAST_ALUMN = Column(Integer)
    created_at = Column(DateTime, server_default=func.now())


class DimHorario(Base):
    __tablename__ = 'dim_horario'

    HORARIO = Column(String, primary_key=True, index=True)
    PERIODO_FRANJA = Column(String)
    TURNO_1 = Column(String)
    TURNO_2 = Column(String)
    TURNO_3 = Column(String)
    TURNO_4 = Column(String)
    created_at = Column(DateTime, server_default=func.now())


class DimAulas(Base):
    __tablename__ = 'dim_aulas'

    PERIODO = Column(Integer, primary_key=True, index=True)
    SEDE = Column(String, primary_key=True, index=True)
    N_AULA = Column(String, primary_key=True, index=True)
    AFORO = Column(Integer)
    created_at = Column(DateTime, server_default=func.now())


class DimSedes(Base):
    __tablename__ = 'dim_sedes'

    SEDE = Column(String, primary_key=True, index=True)
    REGION = Column(String)
    LINEA_DE_NEGOCIO = Column(String)
    created_at = Column(DateTime, server_default=func.now())


class DimRewardsSedes(Base):
    __tablename__ = 'dim_rewards_sedes'

    SEDE = Column(String, primary_key=True, index=True)
    N_AULA = Column(String, primary_key=True, index=True)
    NIVEL = Column(String, primary_key=True, index=True)
    REWARD = Column(Integer)
    created_at = Column(DateTime, server_default=func.now())


class DimVacAcad(Base):
    __tablename__ = 'dim_vac_acad'

    PERIODO = Column(Integer, primary_key=True, index=True)
    LINEA_DE_NEGOCIO = Column(String, primary_key=True, index=True)
    NIVEL = Column(String, primary_key=True, index=True)
    VAC_ACAD_ESTANDAR = Column(Integer)
    VAC_ACAD_MAX = Column(Integer)
    created_at = Column(DateTime, server_default=func.now())


class DimHorariosAtencion(Base):
    __tablename__ = 'dim_horarios_atencion'

    PERIODO = Column(Integer, primary_key=True, index=True)
    SEDE = Column(String, primary_key=True, index=True)
    PERIODO_FRANJA = Column(String, primary_key=True, index=True)
    FRANJA = Column(String, primary_key=True, index=True)
    created_at = Column(DateTime, server_default=func.now())


class DimCursos(Base):
    __tablename__ = 'dim_cursos'

    FRECUENCIA = Column(String)
    NIVEL = Column(String)
    CURSO_ANTERIOR = Column(String, primary_key=True, index=True)
    CURSO_ACTUAL = Column(String)
    DURACION = Column(String)
    created_at = Column(DateTime, server_default=func.now())


class DimDias(Base):
    __tablename__ = 'dim_dias'

    FRECUENCIA = Column(String, primary_key=True, index=True)
    DIA = Column(String, primary_key=True, index=True)
    created_at = Column(DateTime, server_default=func.now())


class DimDiasTurnos(Base):
    __tablename__ = 'dim_dias_turnos'
    DIA = Column(String, primary_key=True, index=True)
    TURNO = Column(String, primary_key=True, index=True)
    created_at = Column(DateTime, server_default=func.now())

class FactProvicional(Base):
    __tablename__ = 'fact_provicional'

    PERIODO = Column(Integer, primary_key=True, index=True)
    SEDE = Column(String, primary_key=True, index=True)
    CODIGO_DE_CURSO = Column(String, primary_key=True, index=True)
    HORARIO = Column(String, primary_key=True, index=True)
    AULA = Column(String, primary_key=True, index=True)
    AULA_PROVICIONAL = Column(String)
    created_at = Column(DateTime, server_default=func.now())