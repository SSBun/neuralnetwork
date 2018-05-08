import os


class Config:

    SECRET_KEY = os.environ.get('SERET_KEY') or 'hard to guess string'
    SQLALCHEMY_COMMIT_ON_TEARDOWN = True

    FLASKY_MAIL_SENDER = 'SSbun <1150183856@qq.com>'
    FLASKY_ADMIN = 'caishilin@yahoo.com'

    SQLALCHEMY_DATABASE_URI = 'mysql+mysqldb://root:Bz550527534@172.96.221.177/timeManager'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    @staticmethod
    def init_app(app):
        pass


class DevelopmentConfig(Config):
    DEBUG = True
    MAIL_SERVER = 'smtp.qq.com'
    MAIL_PORT = 465
    MAIL_USE_SSL = True
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME') or '1150183856@qq.com'
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD') or 'SSBun550527534'


class TestingConfig(Config):
    TESTING = True


class ProductionConfig(Config):
    pass


config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
