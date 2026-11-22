import pickle
import zlib

from sklearn.ensemble import RandomForestClassifier
from sqlalchemy import Column, Engine, Integer, LargeBinary, String, Text
from sqlalchemy.dialects.mysql import LONGBLOB
from sqlalchemy.orm import Session, declarative_base

from mailsort.base.database import (
    MYSQL_TABLE_ARGS,
    SHORT_STRING_LENGTH,
    DatabaseTemplate,
    rename_legacy_table,
)

Base = declarative_base()

# Pickled random forest models easily exceed the 16 MB limit of MEDIUMBLOB
LongBinary = LargeBinary().with_variant(LONGBLOB(), "mysql")

# MySQL rejects statements larger than max_allowed_packet (64 MB by default and
# at most 1 GB), while a pickled calibrated random forest can be several hundred
# MB. So the compressed pickle is split into chunks, stored as individual rows.
MODEL_CHUNK_SIZE = 1024 * 1024


class MachineLearningFolders(Base):
    __tablename__ = "ml_folders"
    __table_args__ = MYSQL_TABLE_ARGS
    id = Column(Integer, primary_key=True)
    folder_id = Column(String(SHORT_STRING_LENGTH))
    random_forest = Column(LongBinary)
    user_id = Column(Integer)


class MachineLearningModelChunks(Base):
    __tablename__ = "ml_folder_chunks"
    __table_args__ = MYSQL_TABLE_ARGS
    id = Column(Integer, primary_key=True)
    folder_id = Column(String(SHORT_STRING_LENGTH))
    chunk_index = Column(Integer)
    data = Column(LongBinary)
    user_id = Column(Integer)


class MachineLearningFeatures(Base):
    __tablename__ = "ml_features"
    __table_args__ = MYSQL_TABLE_ARGS
    id = Column(Integer, primary_key=True)
    # Features are derived from the recipients, which can exceed 255 characters
    feature = Column(Text)
    user_id = Column(Integer)


class MachineLearningDatabase(DatabaseTemplate):
    def store_models(
        self,
        model_dict: dict[str, RandomForestClassifier],
        feature_lst: list[str],
        user_id: int = 1,
        commit: bool = True,
    ) -> None:
        """
        Store machine learning models in database

        Args:
            model_dict (dict): dictionary containing the machine learning models
            feature_lst (list): list of features the machine learning models were trained on
            user_id (int): database user id
            commit (boolean): boolean flag to write to the database
        """
        feature_filtered_lst = [
            feature for feature in feature_lst if feature != "email_id"
        ]
        folder_stored_lst = self._get_folders(user_id=user_id)
        feature_stored_lst = self.get_features(user_id=user_id)
        model_dict_new = {
            k: v for k, v in model_dict.items() if k not in folder_stored_lst
        }
        model_dict_update = {
            k: v for k, v in model_dict.items() if k in folder_stored_lst
        }
        model_delete_lst = [
            folder for folder in folder_stored_lst if folder not in model_dict
        ]
        feature_new_lst = [
            feature
            for feature in feature_filtered_lst
            if feature not in feature_stored_lst
        ]
        feature_remove_lst = [
            feature
            for feature in feature_stored_lst
            if feature not in feature_filtered_lst
        ]
        if len(model_dict_new) > 0:
            self._session.add_all(
                [
                    MachineLearningFolders(folder_id=k, user_id=user_id)
                    for k in model_dict_new
                ]
            )
            for k, v in model_dict_new.items():
                self._store_model_chunks(folder=k, model=v, user_id=user_id)
        if len(feature_new_lst) > 0:
            self._session.add_all(
                [
                    MachineLearningFeatures(feature=feature, user_id=user_id)
                    for feature in feature_new_lst
                ]
            )
        if len(model_dict_update) > 0:
            folder_obj_lst = (
                self._session.query(MachineLearningFolders)
                .filter(MachineLearningFolders.user_id == user_id)
                .filter(
                    MachineLearningFolders.folder_id.in_(list(model_dict_update.keys()))
                )
                .all()
            )
            for folder_obj in folder_obj_lst:
                folder_obj.random_forest = None
            self._delete_model_chunks(
                folder_lst=list(model_dict_update.keys()), user_id=user_id
            )
            for k, v in model_dict_update.items():
                self._store_model_chunks(folder=k, model=v, user_id=user_id)
        if len(model_delete_lst) > 0:
            self._session.query(MachineLearningFolders).filter(
                MachineLearningFolders.user_id == user_id
            ).filter(MachineLearningFolders.folder_id.in_(model_delete_lst)).delete()
            self._delete_model_chunks(folder_lst=model_delete_lst, user_id=user_id)
        if len(feature_remove_lst) > 0:
            self._session.query(MachineLearningFeatures).filter(
                MachineLearningFeatures.user_id == user_id
            ).filter(MachineLearningFeatures.feature.in_(feature_remove_lst)).delete()
        if commit:
            self._session.commit()

    def load_models(
        self, user_id: int = 1
    ) -> tuple[dict[str, RandomForestClassifier], list[str]]:
        """
        Load models from database

        Args:
            user_id (int): database user id

        Returns:
            dict, list: machine learning model dictionary and feature list
        """
        folder_obj_lst = (
            self._session.query(MachineLearningFolders)
            .filter(MachineLearningFolders.user_id == user_id)
            .all()
        )
        feature_lst = self.get_features(user_id=user_id)
        return {
            folder_obj.folder_id: (
                # Models stored before the chunked storage was introduced
                pickle.loads(folder_obj.random_forest)
                if folder_obj.random_forest is not None
                else self._load_model_chunks(
                    folder=folder_obj.folder_id, user_id=user_id
                )
            )
            for folder_obj in folder_obj_lst
        }, feature_lst

    def _store_model_chunks(self, folder: str, model, user_id: int = 1) -> None:
        data = zlib.compress(pickle.dumps(model))
        for chunk_index, start in enumerate(range(0, len(data), MODEL_CHUNK_SIZE)):
            self._session.add(
                MachineLearningModelChunks(
                    folder_id=folder,
                    chunk_index=chunk_index,
                    data=data[start : start + MODEL_CHUNK_SIZE],
                    user_id=user_id,
                )
            )
            # Flush each chunk individually, so the chunks are not combined in
            # a single multi-row INSERT statement exceeding max_allowed_packet
            self._session.flush()

    def _load_model_chunks(self, folder: str, user_id: int = 1):
        chunk_lst = (
            self._session.query(MachineLearningModelChunks.data)
            .filter(MachineLearningModelChunks.user_id == user_id)
            .filter(MachineLearningModelChunks.folder_id == folder)
            .order_by(MachineLearningModelChunks.chunk_index)
            .all()
        )
        return pickle.loads(zlib.decompress(b"".join(chunk[0] for chunk in chunk_lst)))

    def _delete_model_chunks(self, folder_lst: list[str], user_id: int = 1) -> None:
        self._session.query(MachineLearningModelChunks).filter(
            MachineLearningModelChunks.user_id == user_id
        ).filter(MachineLearningModelChunks.folder_id.in_(folder_lst)).delete()

    def get_folders(self, user_id: int = 1) -> list[str]:
        """
        List the folders/folders a machine learning model has been trained for.

        Args:
            user_id (int): database user id

        Returns:
            list: folder names with a stored model - does not unpickle the models
        """
        return self._get_folders(user_id=user_id)

    def _get_folders(self, user_id: int = 1) -> list[str]:
        return [
            folder[0]
            for folder in self._session.query(MachineLearningFolders.folder_id)
            .filter(MachineLearningFolders.user_id == user_id)
            .all()
        ]

    def get_features(self, user_id: int = 1) -> list[str]:
        return [
            feature_obj.feature
            for feature_obj in (
                self._session.query(MachineLearningFeatures)
                .filter(MachineLearningFeatures.user_id == user_id)
                .all()
            )
        ]


def get_machine_learning_database(
    engine: Engine, session: Session
) -> MachineLearningDatabase:
    for old_table, new_table in (
        ("ml_labels", "ml_folders"),
        ("ml_label_chunks", "ml_folder_chunks"),
    ):
        rename_legacy_table(
            engine=engine,
            old_table=old_table,
            new_table=new_table,
            old_column="label_id",
            new_column="folder_id",
        )
    Base.metadata.create_all(engine)
    return MachineLearningDatabase(session=session)
